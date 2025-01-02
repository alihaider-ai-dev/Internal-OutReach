import streamlit as st
import pandas as pd
import json
from typing import Optional, Dict, Any, Union
from datetime import datetime
from anthropic import Anthropic
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import os
from dotenv import load_dotenv
from variables import usecases
load_dotenv()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'website_content' not in st.session_state:
    st.session_state.website_content = None

class UseCaseAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.MODEL_NAME = "claude-3-5-sonnet-20241022"

    def get_usecase_schema(self) -> Dict:
        """Define the schema for use case analysis"""
        return {
            "type": "object",
            "properties": {
                "use_cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the use case"
                            },
                            "domain": {
                                "type": "string",
                                "description": "Business domain of the use case"
                            },
                            "problem": {
                                "type": "string",
                                "description": "Detailed Description of the problem being addressed"
                            },
                            "ai_solution": {
                                "type": "string",
                                "description": "Deatiled Proposed AI-based solution"
                            },
                            "feasibility": {
                                "type": "object",
                                "properties": {
                                    "score": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 10
                                    },
                                    "rationale": {
                                        "type": "string"
                                    }
                                }
                            },
                            "efficiency_gains": {
                                "type": "object",
                                "properties": {
                                    "quantitative": {
                                        "type": "string",
                                        "description": "Measurable efficiency improvements"
                                    },
                                    "qualitative": {
                                        "type": "string",
                                        "description": "Qualitative benefits"
                                    }
                                }
                            },
                            "requirements": {
                                "type": "object",
                                "properties": {
                                    "data": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "technology": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["name", "domain", "problem", "ai_solution", "feasibility", 
                                   "efficiency_gains", "requirements"]
                    }
                }
            },
            "required": ["use_cases"]
        }

    def analyze_usecases(self, website_content: str,number_of_usecases:int, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze website content to identify and structure use cases"""
        tools = [{
            "name": "get_usecases",
            "description": "Analyze the website content and extract structured use cases",
            "input_schema": self.get_usecase_schema()
        }]

        system_prompt = f"""
        You are a use case analysis expert. Your task is to:
        
        1. Analyze the provided website content
        2. Identify prominent use cases across different domains
        3. Structure each use case with detailed problem description, AI solution approach,
           feasibility assessment, efficiency gains, and implementation requirements
        4. Focus on practical, implementable solutions
        5. Provide quantitative metrics where possible
        6. Consider technical feasibility and data requirements
        7. Generate Only {number_of_usecases} usecsaes.
    
        """

        query = f"""
        <website_content>
            {website_content}
        </website_content>
        
        {f'<additional_context>{context}</additional_context>' if context else ''}
        
        Analyze the content to identify and structure key use cases.
        Return Only {number_of_usecases} usecases.
        
        """

        try:
            response = self.client.messages.create(
                model=self.MODEL_NAME,
                max_tokens=4096,
                tools=tools,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )

            # Extract the tool use response
            for content in response.content:
                if content.type == "tool_use" and content.name == "get_usecases":
                    return content.input

            raise Exception("No structured use case analysis found in the response")

        except Exception as e:
            raise Exception(f"Error analyzing use cases: {str(e)}")

def get_text_using_scrape_do(url: str, max_pages: int) -> str:
    """Get text content from website using scrape.do service"""
    try:
        urls = get_all_links(url)
        urls = urls[:min(max_pages, len(urls))]
        urls.append(url)
        urls = urls[::-1]
        
        scrape_do_token = os.getenv("SCRAPE_DO_TOKEN")
        context = ''
        
        for url in urls:
            target_url = requests.utils.quote(url)
            api_url = f"http://api.scrape.do?token={scrape_do_token}&url={target_url}"
            response = requests.request("GET", api_url)
            context += extract_text_from_html(response.text)
        
        return context
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return ""

def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    text = soup.get_text()
    return ' '.join(text.split())

def get_all_links(url: str) -> list:
    """Get all links from a webpage"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('#'):
            full_url = urljoin(url, href)
            links.append(full_url)
    return links

def json_to_usecase_dataframe(json_data: Union[Dict, str]) -> pd.DataFrame:
    """Convert JSON use case data to pandas DataFrame"""
    def format_requirements(req: Dict) -> str:
        try:
            data = '; '.join(req['data'])
            tech = '; '.join(req['technology'])
            return f"Data: [{data}]\nTechnology: [{tech}]"
        except KeyError as e:
            raise ValueError(f"Missing required field in requirements: {e}")

    def format_efficiency_gains(gains: Dict) -> str:
        try:
            return f"Quantitative: {gains['quantitative']}\nQualitative: {gains['qualitative']}"
        except KeyError as e:
            raise ValueError(f"Missing required field in efficiency gains: {e}")

    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
            
        if not isinstance(json_data, dict) or 'use_cases' not in json_data:
            raise ValueError("Input JSON must contain 'use_cases' key")
            
        flattened_data = []

        for use_case in json_data['use_cases']:
            try:
                flattened_case = {
                    'Name': use_case['name'],
                    'Domain': use_case['domain'],
                    'Problem': use_case['problem'],
                    'AI Solution': use_case['ai_solution'],
                    'Feasibility Score': use_case['feasibility']['score'],
                    'Feasibility Rationale': use_case['feasibility']['rationale'],
                    'Efficiency Gains': format_efficiency_gains(use_case['efficiency_gains']),
                    'Requirements': format_requirements(use_case['requirements'])
                }
                flattened_data.append(flattened_case)
            except KeyError as e:
                raise ValueError(f"Missing required field in use case: {e}")

        df = pd.DataFrame(flattened_data)
        return df

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")
    except Exception as e:
        raise ValueError(f"Error processing JSON data: {e}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Use Case Analyzer", page_icon="🔍", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .st-expander {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Website Use Case Analyzer")
    st.markdown("""
    <div style='background-color: black; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        This application analyzes websites to identify and structure potential use cases for AI implementation.
        Enter a website URL below to get started with the analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_pages = st.slider("Maximum pages to analyze", 1, 10, 3)
        st.info("Analyzing more pages will take longer but provide more comprehensive results.")
    
    # Main form
    with st.form("analysis_form"):
        
            website_url = st.text_input("🌐 Website URL", placeholder="https://example.com")
            number_of_usecases = st.slider("Number of Usecases to Generate", 1, 10)
            submit_button = st.form_submit_button("🔍 Analyze Website",use_container_width=True)
    
    if submit_button and website_url:
        try:
                with st.spinner("🌐 Fetching website content..."):
                    st.session_state.website_content = get_text_using_scrape_do(website_url, max_pages)
            
                with st.spinner("🤖 Analyzing use cases..."):
                        analyzer = UseCaseAnalyzer()
                        st.session_state.analysis_results = analyzer.analyze_usecases( website_content=st.session_state.website_content, number_of_usecases,context=usecases )
                
                # Convert results to DataFrame
                df = json_to_usecase_dataframe(st.session_state.analysis_results)
                
                # Display success message
                st.success("Analysis completed successfully!")
                
                # Display raw DataFrame
                with st.expander("View Raw Data"):
                    st.dataframe(df, use_container_width=True)
                
                # Display detailed results
                st.markdown("### Detailed Use Cases")
                for idx, row in df.iterrows():
                    with st.expander(f"📌 {row['Name']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📋 Overview")
                            st.info(f"**Domain:** {row['Domain']}")
                            st.write(f"**Problem:** {row['Problem']}")
                            st.write(f"**AI Solution:** {row['AI Solution']}")
                        
                        with col2:
                            st.markdown("#### 🎯 Details")
                            st.metric("Feasibility Score", f"{row['Feasibility Score']}/10")
                            st.write(f"**Rationale:** {row['Feasibility Rationale']}")
                            st.write("**📈 Efficiency Gains:**")
                            st.code(row['Efficiency Gains'])
                            st.write("**⚙️ Requirements:**")
                            st.code(row['Requirements'])
                
                with st.expander("View as Table"):
                    st.dataframe(df,use_container_width=True)

                # Download options
                st.markdown("### Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"use_case_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json.dumps(st.session_state.analysis_results, indent=2),
                        file_name=f"use_case_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json',use_container_width=True
                    )
                
                # Reset button
                if st.button("🔄 Start New Analysis"):
                    del st.session_state['website_content']
                    del st.session_state['analysis_results']
                    st.rerun()
                    
            
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
    elif submit_button:
        if not website_url:
            st.warning("⚠️ Please enter a website URL")

if __name__ == "__main__":
    main()
