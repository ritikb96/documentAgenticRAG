import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from pydantic import BaseModel,EmailStr,ValidationError
from typing import Literal, TypedDict, Optional, Annotated
from langgraph.store.redis import RedisStore
from langgraph.store.base import IndexConfig
from langgraph.checkpoint.redis import RedisSaver
from app.services.redis_handler import get_redis_store
from datetime import date,time,datetime,timezone
from app.services.supabase_handler import insert_bookingData
from app.services.email import notify_admin_booking

# Globals for tools
retriever = None
tavily = None

@tool
def rag_search_tool(query: str) -> list:
    """Search Qdrant-based knowledge base for relevant chunks."""
    return retriever.invoke(query)

@tool
def web_search_tool(query: str) -> str:
    """Fetch up-to-date web information using TavilySearch."""
    res = tavily.invoke({"query": query})
    return "\n".join(result["content"] for result in res["results"])

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    rag: Optional[str]
    web: Optional[str]

class RouteDecision(BaseModel):
    route: Literal["rag", "answer","web","end","interview"]
    reply: Optional[str] = None

class RagJudge(BaseModel):
    sufficient: bool

class booking_data(BaseModel):
    id:str
    full_name:str
    email:EmailStr
    interview_date:date
    time:time
    created_at:datetime



class DocuRAGAgent:
    def __init__(self, thread_id: str = "thread-12"):
        global retriever, tavily
        load_dotenv()
        self.thread_id = thread_id
        self.LLM_MODEL = os.getenv("LLM_MODEL")

        self.llm_router = ChatOpenAI(model=self.LLM_MODEL, temperature=0).with_structured_output(RouteDecision)
        self.llm_judge = ChatOpenAI(model=self.LLM_MODEL, temperature=0).with_structured_output(RagJudge)
        self.llm_answer = ChatOpenAI(model=self.LLM_MODEL, temperature=0.7)

        tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=3, topic="general")
        retriever = self._init_vectorstore()

        self.agent = self._build_agent()

    def _init_vectorstore(self):
        client = QdrantClient(url="http://localhost:6333")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_db = QdrantVectorStore(
            client=client,
            collection_name="docuRAG-embeddings",
            embedding=embedding_model,
        )
        return vector_db.as_retriever(search_kwargs={"k": 3})

    #setup all the nodes of the graph

    def _router_node(self, state: AgentState) -> AgentState:
        q = state["messages"][-1].content
        decision: RouteDecision = self.llm_router.invoke([
            SystemMessage(content="You are a routing agent. For all questions seeking factual, knowledge-based answers \
            (e.g., 'Where did x live?','What is X?') Route to 'rag'.  \
            If the path provides sufficient information to respond directly to the user, or chit-chat route to 'answer'.  \
            For interview booking, route to 'interview'. Route to 'end' only if the  \
            Use 'web' if the question is about current events, latest news, trending topics, or things  \
            that may have changed recently (like today's weather, or latest election news).\
            user says goodbye or wants to end." ),
            HumanMessage(content=q)
        ])

        print(f"Router decision: {decision.route}")
        new_state = {**state, "route": decision.route}
        if decision.route == "end":
            new_state["messages"].append(AIMessage(content=decision.reply or "Goodbye!"))
        return new_state

    def _rag_node(self, state: AgentState) -> AgentState:
        q = state["messages"][-1].content
        
        # Get list of Document objects
        docs = rag_search_tool.invoke({"query": q})
        
        # Extract content from top 3 documents
        context = "\n\n".join(doc.page_content for doc in docs[:3])
        print(f"🔍 Retrieved: {[doc.page_content for doc in docs]}")


        
        # Build prompt using the extracted context
        prompt = f"Question: {q}\n\nContext:\n{context}"
        
        # Let the judge decide if context is sufficient
        verdict: RagJudge = self.llm_judge.invoke([HumanMessage(content=prompt)])
        
        return {
            **state,
            "rag": docs,  # Store full document list in state
            "route": "answer" if verdict.sufficient else "web"
        }

    def _web_node(self, state: AgentState) -> AgentState:
        q = state["messages"][-1].content
        web_snippets = web_search_tool.invoke({"query": q})
        return {**state, "web": web_snippets, "route": "answer"}

    @staticmethod
    def from_router(state: AgentState) -> str:
        return state.get("route", "end")

    @staticmethod
    def after_rag(state: AgentState) -> str:
        return state.get("route", "answer")

    


    def _answer_node(self, state: AgentState) -> AgentState:
        user_query = state["messages"][-1].content
        thread_id = self.thread_id
        namespace = ("memories", thread_id)

        # Search memory
        past_memories = self.redis_store.search(namespace, query=user_query)
        memory_context = "\n".join([doc.value["data"] for doc in past_memories])

        #Combine context sources
        context = ""
        if memory_context:
            context += f"Memory:\n{memory_context}\n\n"
        if state.get("rag"):
            context += f"Knowledge:\n{state['rag']}\n\n"
        if state.get("web"):
            context += f"Web:\n{state['web']}\n\n"

        #Generate answer using LLM
        final_answer = self.llm_answer.invoke([
            ("system", f"You are an assistant. Use the provided context to answer.\n{context}"),
            ("user", user_query)
        ])

        # 4. Always store the full Q&A pair as memory
        combined_memory = f"User: {user_query}\nAssistant: {final_answer.content}"
        self.redis_store.put(namespace, str(uuid.uuid4()), {"data": combined_memory})
        
        return {
            **state,
            "messages": state["messages"] + [final_answer],
            "route": "end"
        }
    
    def _booking_node(self, state: AgentState) -> AgentState:
        user_query = state["messages"][-1].content
        thread_id = self.thread_id
        namespace = ("memories", thread_id)

        #booking llm
        booking_llm = ChatOpenAI(model=self.LLM_MODEL, temperature=0).with_structured_output(booking_data)

        try:
            # Extract booking details from the user query
            booking_info: booking_data = booking_llm.invoke([
                SystemMessage(content="Extract full_name, email, interview_date (YYYY-MM-DD), and time (HH:MM) from the user message."),
                HumanMessage(content=user_query)
            ])

            # Create full booking_data object
            booking = booking_data(
                id= str(uuid.uuid4()),
                full_name=booking_info.full_name,
                email=booking_info.email,
                interview_date=booking_info.interview_date,
                time=booking_info.time,
                created_at=datetime.now(timezone.utc).isoformat()
            )

            # Store booking and send notification
            insert_bookingData(booking)
            notify_admin_booking(booking)

            confirmation_msg = AIMessage(content=f"Thanks {booking.full_name}! Your interview is booked on {booking.interview_date} at {booking.time}. We've sent a confirmation email.")

        except ValidationError as e:
            # Handle  errors
            confirmation_msg = AIMessage(content="Sorry, I couldn't extract all the booking details correctly. Please provide your full name, email, interview date (YYYY-MM-DD), and time (HH:MM).")

       
        return {
            **state,
            "messages": state["messages"] + [confirmation_msg],
            "route": "end"
        }


    def _build_agent(self):
        REDIS_URI = "redis://localhost:6379"

        with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
            checkpointer.setup()

        self.redis_store = get_redis_store()

        agent_graph = StateGraph(AgentState)
        
        # Add nodes
        agent_graph.add_node("router", self._router_node)
        agent_graph.add_node("rag_lookup", self._rag_node)
        agent_graph.add_node("web_search", self._web_node)
        agent_graph.add_node("answer", self._answer_node)
        agent_graph.add_node("book_interview", self._booking_node)

        # Set entry point
        agent_graph.set_entry_point("router")

        # Routing logic: Always ends in "answer"
        agent_graph.add_conditional_edges("router", self.from_router, {
            "rag": "rag_lookup",
            "answer": "answer",    
            "interview":"book_interview",
            "web":"web_search",
            "end":"answer"

        })

        # After rag lookup: go to web or answer
        agent_graph.add_conditional_edges("rag_lookup", self.after_rag, {
            "web": "web_search",
            "answer": "answer"
        })

        # Web always leads to answer
        agent_graph.add_edge("web_search", "answer")

        agent_graph.add_edge("book_interview",END)

        # Final output from answer
        agent_graph.add_edge("answer", END)

        self.agent_graph = agent_graph
        return agent_graph.compile(checkpointer=checkpointer, store=self.redis_store)


    def run(self, query: str, thread_id: str = "thread-12") -> str:
        self.thread_id = thread_id
        state = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": self.thread_id}}
        result = self.agent.invoke(state, config)
        return result["messages"][-1].content
