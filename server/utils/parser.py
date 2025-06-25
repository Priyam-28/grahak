from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **Format:** Return the information in a clean, structured format. "
    "3. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "4. **Empty Response:** If no information matches the description, return 'No relevant products found in this section.'"
    "5. **Direct Data Only:** Your output should contain only the requested product data."
)

model = OllamaLLM(model="llama3.2")

def parse_with_ollama(dom_chunks, parse_description):
    """Parse DOM chunks with improved error handling and result formatting"""
    try:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        parsed_results = []
        successful_chunks = 0

        for i, chunk in enumerate(dom_chunks, start=1):
            try:
                logger.info(f"Processing chunk {i} of {len(dom_chunks)}")
                
                response = chain.invoke(
                    {"dom_content": chunk, "parse_description": parse_description}
                )
                
                # Clean up the response
                if isinstance(response, str):
                    cleaned_response = response.strip()
                    if cleaned_response and cleaned_response != "No relevant products found in this section.":
                        parsed_results.append(cleaned_response)
                        successful_chunks += 1
                        logger.info(f"Successfully parsed chunk {i}")
                    else:
                        logger.info(f"No relevant content found in chunk {i}")
                else:
                    logger.warning(f"Unexpected response type from chunk {i}: {type(response)}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        logger.info(f"Completed processing. {successful_chunks} out of {len(dom_chunks)} chunks contained relevant data.")

        if parsed_results:
            # Join results with clear separators
            final_result = "\n\n---\n\n".join(parsed_results)
            return final_result
        else:
            return "No relevant products found on this website."
            
    except Exception as e:
        logger.error(f"Error in parse_with_ollama: {str(e)}")
        return f"Error during parsing: {str(e)}"