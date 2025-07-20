from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
import tempfile
from pathlib import Path
from io import BytesIO

class UploadService:
    def __init__(self):
        accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            do_ocr=True,
            do_table_structure=True
        )
        pipeline_options.table_structure_options.do_cell_matching = True

        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    async def parse_file(self, filename: str, content: bytes) -> str:
        """Parse uploaded file content to text or markdown."""
        if filename.endswith('.txt'):
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ValueError("Text file must be UTF-8 encoded") from e

        elif filename.endswith('.pdf'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(content)
                    tmp_file_path = Path(tmp_file.name)

                doc = self.docling_converter.convert(tmp_file_path, {"input_format": "pdf"}).document
                print("Document was successfully parsed")
                return doc.export_to_markdown()
            except Exception as e:
                raise RuntimeError(f"PDF parsing failed: {str(e)}") from e

        else:
            raise ValueError("Only .pdf and .txt files are supported")
