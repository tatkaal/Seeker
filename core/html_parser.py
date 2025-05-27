from bs4 import BeautifulSoup
import re

class HTMLParser:
    @staticmethod
    def clean_html(html_content: str) -> str:
        """Cleans HTML and extracts raw text."""
        if not html_content or not isinstance(html_content, str):
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text