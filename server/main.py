import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import time
from groq import Groq
import base64
from elevenlabs.client import ElevenLabs
import io
import speech_recognition as sr
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Page config
st.set_page_config(
    page_title="Voice AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'products' not in st.session_state:
    st.session_state.products = []

# Sidebar for API Keys
st.sidebar.title("🔧 Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password", help="Enter your ElevenLabs API key")

# Main title
st.title("🛒 Voice AI Shopping Assistant")
st.markdown("*Speak or type to find products on Amazon with AI assistance*")

class AmazonScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        }
    
    def search_products(self, query, max_results=5):
        """Search Amazon for products"""
        try:
            search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            products = []
            
            # Find product containers
            product_containers = soup.find_all('div', {'data-component-type': 's-search-result'})
            
            for container in product_containers[:max_results]:
                try:
                    # Extract product data
                    title_elem = container.find('h2', class_='a-size-mini')
                    if title_elem:
                        title = title_elem.get_text().strip()
                    else:
                        continue
                    
                    # Price
                    price_elem = container.find('span', class_='a-price-whole')
                    price = "N/A"
                    if price_elem:
                        price = f"${price_elem.get_text().strip()}"
                    
                    # Rating
                    rating_elem = container.find('span', class_='a-icon-alt')
                    rating = "N/A"
                    if rating_elem:
                        rating_text = rating_elem.get_text()
                        if 'out of' in rating_text:
                            rating = rating_text.split(' out of')[0]
                    
                    # Image
                    img_elem = container.find('img', class_='s-image')
                    image_url = ""
                    if img_elem and img_elem.get('src'):
                        image_url = img_elem.get('src')
                    
                    # Link
                    link_elem = container.find('h2', class_='a-size-mini').find('a') if container.find('h2', class_='a-size-mini') else None
                    product_url = ""
                    if link_elem and link_elem.get('href'):
                        product_url = f"https://www.amazon.com{link_elem.get('href')}"
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'rating': rating,
                        'image_url': image_url,
                        'product_url': product_url
                    })
                    
                except Exception as e:
                    continue
            
            return products
            
        except Exception as e:
            st.error(f"Error scraping Amazon: {str(e)}")
            return []

class VoiceShoppingAgent:
    def __init__(self, groq_api_key):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.scraper = AmazonScraper()
    
    def parse_intent(self, user_input):
        """Parse user intent and extract search query"""
        if not self.groq_client:
            return user_input
        
        try:
            prompt = f"""
            You are a shopping assistant. Parse the user's request and extract the main product they want to search for.
            Return only the search term, nothing else.
            
            User input: "{user_input}"
            
            Examples:
            "I want headphones for gym" -> "gym headphones"
            "Looking for a laptop under 1000 dollars" -> "laptop under 1000"
            "Need wireless earbuds" -> "wireless earbuds"
            
            Search term:"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=50,
                temperature=0.1,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return user_input
    
    def generate_response(self, user_input, products):
        """Generate conversational response about products"""
        if not self.groq_client:
            return "I found some products for you!"
        
        try:
            products_text = "\n".join([
                f"- {p['title'][:100]}... (Price: {p['price']}, Rating: {p['rating']})"
                for p in products[:3]
            ])
            
            prompt = f"""
            You are a helpful shopping assistant. The user asked: "{user_input}"
            
            Here are the products I found:
            {products_text}
            
            Provide a brief, conversational response (2-3 sentences) highlighting the key findings and asking if they need more specific help.
            Be friendly and helpful.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return "I found some products for you! Take a look at the results below."

def text_to_speech(text, api_key):
    """Convert text to speech using ElevenLabs"""
    if not api_key:
        return None

    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.generate(
            text=text,
            voice="Bella",
            model="eleven_monolingual_v1"
        )
        return audio
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def speech_to_text(audio_bytes):
    """Convert speech to text"""
    try:
        r = sr.Recognizer()
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Convert to text
        with sr.AudioFile(tmp_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        
        # Clean up
        os.unlink(tmp_file_path)
        return text
        
    except Exception as e:
        st.error(f"Error converting speech to text: {str(e)}")
        return None

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Voice input
    st.markdown("**🎤 Voice Input:**")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )
    
    # Text input
    user_input = st.text_input("🔤 Or type your request:", placeholder="e.g., I need wireless headphones under $100")
    
    # Process voice input
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("🔍 Convert Voice to Text"):
            with st.spinner("Converting speech to text..."):
                voice_text = speech_to_text(audio_bytes)
                if voice_text:
                    user_input = voice_text
                    st.success(f"Voice input: {voice_text}")

# Process search
if user_input and (groq_api_key or st.sidebar.checkbox("Skip AI (Direct Search)")):
    with st.spinner("Searching for products..."):
        # Initialize agent
        agent = VoiceShoppingAgent(groq_api_key)
        
        # Parse intent and search
        search_query = agent.parse_intent(user_input) if groq_api_key else user_input
        products = agent.scraper.search_products(search_query)
        
        if products:
            st.session_state.products = products
            
            # Generate AI response
            ai_response = agent.generate_response(user_input, products)
            st.session_state.messages.append({"user": user_input, "assistant": ai_response})
            
            # Display response
            st.success(ai_response)
            
            # Text-to-speech
            if elevenlabs_api_key:
                with st.spinner("Generating speech..."):
                    audio_data = text_to_speech(ai_response, elevenlabs_api_key)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
        else:
            st.warning("No products found. Try a different search term.")

with col2:
    st.subheader("📊 Quick Stats")
    if st.session_state.products:
        st.metric("Products Found", len(st.session_state.products))
        avg_rating = sum([float(p['rating'].split()[0]) for p in st.session_state.products if p['rating'] != 'N/A' and p['rating'].split()[0].replace('.', '').isdigit()]) / len([p for p in st.session_state.products if p['rating'] != 'N/A'])
        if avg_rating:
            st.metric("Avg Rating", f"{avg_rating:.1f}⭐")

# Display products
if st.session_state.products:
    st.subheader("🛍️ Product Results")
    
    for i, product in enumerate(st.session_state.products):
        with st.expander(f"📦 {product['title'][:80]}..." if len(product['title']) > 80 else f"📦 {product['title']}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if product['image_url']:
                    st.image(product['image_url'], width=150)
            
            with col2:
                st.write(f"**Price:** {product['price']}")
                st.write(f"**Rating:** {product['rating']}")
                if product['product_url']:
                    st.markdown(f"[🔗 View on Amazon]({product['product_url']})")

# Chat history
if st.session_state.messages:
    st.subheader("💭 Conversation History")
    for msg in st.session_state.messages[-3:]:  # Show last 3 conversations
        st.write(f"**You:** {msg['user']}")
        st.write(f"**Assistant:** {msg['assistant']}")
        st.divider()

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, Groq API, ElevenLabs, and Amazon Web Scraping*")

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    1. **Setup:** Enter your Groq and ElevenLabs API keys in the sidebar
    2. **Voice Input:** Click the microphone button to record your request
    3. **Text Input:** Or type your product search request
    4. **AI Processing:** The assistant will understand your intent and search Amazon
    5. **Results:** View products with images, prices, and ratings
    6. **Voice Output:** Listen to the AI response (if ElevenLabs key provided)
    
    **Example queries:**
    - "I need wireless headphones under $100"
    - "Looking for a gaming laptop"
    - "Show me running shoes for women"
    """)

# API Key help
with st.expander("🔑 API Key Setup"):
    st.markdown("""
    **Groq API Key:**
    1. Visit [https://console.groq.com/](https://console.groq.com/)
    2. Sign up/login and create an API key
    3. Paste it in the sidebar
    
    **ElevenLabs API Key:**
    1. Visit [https://elevenlabs.io/](https://elevenlabs.io/)
    2. Sign up and get your API key from settings
    3. Paste it in the sidebar
    
    *Note: ElevenLabs key is optional - the app works without voice output*
    """)