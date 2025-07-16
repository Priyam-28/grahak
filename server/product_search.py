from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tabulate import tabulate
from pymongo import MongoClient
import os, json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import hashlib
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from typing import Tuple, Set
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
import pickle
import logging
from dataclasses import dataclass, field
from enum import Enum
import random



_user_preferences = {} 
_search_history = defaultdict(list)  
_recommendations_cache = {}  
_wishlist = defaultdict(list)  
_cart = defaultdict(list)
_user_sessions = {}  # email -> session data

@dataclass
class CartItem:
    product_id: int
    quantity: int
    added_at: datetime
    
@dataclass
class UserPreferences:
    favorite_categories: List[str]
    price_range: tuple
    preferred_brands: List[str]
    size_preferences: Dict[str, str]

load_dotenv()
mcp = FastMCP("ProductAgent")

# ---------------- MongoDB setup ---------------- #
def mongo_client() -> MongoClient:
    host = os.getenv("MONGO_HOST", "localhost")
    port = int(os.getenv("MONGO_PORT", 27017))
    user = os.getenv("MONGO_USER", "admin")
    pwd  = os.getenv("MONGO_PASS", "password")
    uri  = f"mongodb://{user}:{pwd}@{host}:{port}/?authSource=admin"
    return MongoClient(uri)

try:
    db = mongo_client()["contoso"]
    # Test connection
    db.command("ping")
    print("[MongoDB] Connected successfully")
except Exception as e:
    print(f"[MongoDB] Connection failed: {e}")
    db = None

# ---------------- Globals ---------------- #
_vectordb = None
_current_user_cache = {}  # email -> user_doc cache

# ---------------- Vector DB Setup ---------------- #
def load_products() -> list[dict]:
    try:
        with open("products.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[WARNING] products.json not found, creating sample data")
        # Create sample products if file doesn't exist
        sample_products = [
        ]
        # Save sample data
        with open("products.json", "w") as f:
            json.dump(sample_products, f, indent=2)
        return sample_products


class AIModelManager:
    """Manages different AI models for various tasks"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.summarizer = None
        self.embeddings_model = None
        self.intent_classifier = None
        self.initialized = False
        
    def initialize_models(self):
        """Initialize AI models lazily"""
        if self.initialized:
            return
            
        try:
            # Sentiment Analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            # Text Summarization
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn")
            
            # Intent Classification
            self.intent_classifier = pipeline("zero-shot-classification",
                                             model="facebook/bart-large-mnli")
            
            self.initialized = True
            print("[AI Models] Initialized successfully")
            
        except Exception as e:
            print(f"[AI Models] Failed to initialize: {e}")
            self.initialized = False

ai_models = AIModelManager()

# ==================== ADVANCED SEMANTIC SEARCH ====================

class SemanticSearchEngine:
    """Advanced semantic search with multiple embedding strategies"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.product_embeddings = None
        self.product_tfidf = None
        self.query_cache = {}
        
    def create_advanced_embeddings(self, products: list[dict]):
        """Create multiple types of embeddings for products"""
        try:
            # TF-IDF embeddings
            product_texts = [self._create_product_text(p) for p in products]
            self.product_tfidf = self.tfidf_vectorizer.fit_transform(product_texts)
            
            # Contextual embeddings (if available)
            if ai_models.initialized and ai_models.embeddings_model:
                self.product_embeddings = self._create_contextual_embeddings(product_texts)
            
            print("[Semantic Search] Advanced embeddings created")
            
        except Exception as e:
            print(f"[Semantic Search] Failed to create embeddings: {e}")
    
    def _create_product_text(self, product: dict) -> str:
        """Create comprehensive text representation of product"""
        text_parts = [
            product.get('name', ''),
            product.get('description', ''),
            product.get('category', ''),
            f"price {product.get('price', 0)}",
            f"rating {product.get('rating', 0)}",
            "in stock" if product.get('inStock', True) else "out of stock"
        ]
        return " ".join(filter(None, text_parts))
    
    def semantic_search(self, query: str, products: list[dict], top_k: int = 10) -> list[dict]:
        """Advanced semantic search with multiple scoring strategies"""
        if query in self.query_cache:
            return self.query_cache[query][:top_k]
        
        try:
            # TF-IDF based search
            query_tfidf = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_tfidf, self.product_tfidf).flatten()
            
            # Combine with other scoring strategies
            final_scores = []
            for i, product in enumerate(products):
                score = tfidf_scores[i]
                
                # Boost based on query-product matching
                score += self._calculate_semantic_boost(query, product)
                
                # Boost based on product popularity
                score += self._calculate_popularity_boost(product)
                
                final_scores.append((product, score))
            
            # Sort by score and cache results
            final_scores.sort(key=lambda x: x[1], reverse=True)
            results = [product for product, score in final_scores]
            
            self.query_cache[query] = results
            return results[:top_k]
            
        except Exception as e:
            print(f"[Semantic Search] Search failed: {e}")
            return products[:top_k]
    
    def _calculate_semantic_boost(self, query: str, product: dict) -> float:
        """Calculate semantic relevance boost"""
        query_lower = query.lower()
        product_text = self._create_product_text(product).lower()
        
        # Exact matches
        if query_lower in product_text:
            return 0.5
        
        # Partial matches
        query_words = set(query_lower.split())
        product_words = set(product_text.split())
        overlap = len(query_words.intersection(product_words))
        
        return min(overlap * 0.1, 0.3)
    
    def _calculate_popularity_boost(self, product: dict) -> float:
        """Calculate popularity-based boost"""
        rating = product.get('rating', 0)
        reviews = product.get('reviews', 0)
        
        rating_boost = rating * 0.05 if rating else 0
        review_boost = min(reviews / 1000, 0.1)
        
        return rating_boost + review_boost

semantic_engine = SemanticSearchEngine()

# ==================== ADVANCED PERSONALIZATION ====================

class PersonalizationEngine:
    """Advanced user personalization with ML"""
    
    def __init__(self):
        self.user_embeddings = {}
        self.product_clusters = {}
        self.collaborative_matrix = None
        
    def create_user_embedding(self, email: str) -> np.ndarray:
        """Create user embedding based on behavior"""
        try:
            behavior = analyze_user_behavior(email)
            searches = _search_history.get(email, [])
            wishlist = _wishlist.get(email, [])
            
            # Create feature vector
            features = []
            
            # Search behavior features
            if searches:
                search_texts = [s['query'] for s in searches[-20:]]  # Last 20 searches
                search_embedding = self._text_to_embedding(" ".join(search_texts))
                features.extend(search_embedding[:50])  # First 50 dimensions
            else:
                features.extend([0] * 50)
            
            # Category preferences
            products = load_products()
            categories = list(set(p['category'] for p in products))
            category_prefs = [0] * len(categories)
            
            for i, cat in enumerate(categories):
                if behavior.get('top_categories', {}).get(cat):
                    category_prefs[i] = behavior['top_categories'][cat]
            
            features.extend(category_prefs)
            
            # Price preferences
            avg_price = behavior.get('avg_price_preference', 0)
            price_features = [
                1 if avg_price < 1000 else 0,
                1 if 1000 <= avg_price < 5000 else 0,
                1 if 5000 <= avg_price < 10000 else 0,
                1 if avg_price >= 10000 else 0
            ]
            features.extend(price_features)
            
            # Behavioral features
            features.extend([
                len(searches) / 100,  # Search frequency
                len(wishlist) / 50,   # Wishlist size
                behavior.get('search_count', 0) / 100  # Total searches
            ])
            
            embedding = np.array(features, dtype=np.float32)
            self.user_embeddings[email] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"[Personalization] Failed to create user embedding: {e}")
            return np.zeros(100)
    
    def _text_to_embedding(self, text: str) -> list:
        """Convert text to embedding using TF-IDF"""
        if not hasattr(self, '_temp_vectorizer'):
            self._temp_vectorizer = TfidfVectorizer(max_features=100)
            # Fit on some sample text
            sample_texts = [p.get('name', '') + ' ' + p.get('description', '') 
                          for p in load_products()[:100]]
            self._temp_vectorizer.fit(sample_texts)
        
        try:
            embedding = self._temp_vectorizer.transform([text]).toarray().flatten()
            return embedding.tolist()
        except:
            return [0] * 100
    
    def find_similar_users(self, email: str, top_k: int = 5) -> list[str]:
        """Find similar users for collaborative filtering"""
        if email not in self.user_embeddings:
            self.create_user_embedding(email)
        
        user_embedding = self.user_embeddings[email]
        similarities = []
        
        for other_email, other_embedding in self.user_embeddings.items():
            if other_email != email:
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((other_email, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [email for email, _ in similarities[:top_k]]

personalization_engine = PersonalizationEngine()

# ==================== INTELLIGENT CHATBOT ====================

class IntentType(Enum):
    SEARCH = "search"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    CART = "cart"
    WISHLIST = "wishlist"
    SUPPORT = "support"
    PRICE_INQUIRY = "price_inquiry"
    AVAILABILITY = "availability"

@dataclass
class ChatContext:
    user_email: str
    conversation_history: list = field(default_factory=list)
    current_intent: IntentType = None
    extracted_entities: dict = field(default_factory=dict)
    session_products: list = field(default_factory=list)

class IntelligentChatbot:
    """Advanced chatbot with intent recognition and context awareness"""
    
    def __init__(self):
        self.contexts = {}
        self.intent_patterns = {
            IntentType.SEARCH: ["search", "find", "show", "looking for", "need"],
            IntentType.RECOMMENDATION: ["recommend", "suggest", "best", "top", "popular"],
            IntentType.COMPARISON: ["compare", "vs", "versus", "difference", "better"],
            IntentType.CART: ["cart", "add to cart", "buy", "purchase"],
            IntentType.WISHLIST: ["wishlist", "save", "bookmark", "like"],
            IntentType.SUPPORT: ["help", "support", "problem", "issue", "complaint"],
            IntentType.PRICE_INQUIRY: ["price", "cost", "cheap", "expensive", "budget"],
            IntentType.AVAILABILITY: ["available", "stock", "in stock", "out of stock"]
        }
    
    def process_message(self, user_email: str, message: str) -> dict:
        """Process user message with advanced NLP"""
        try:
            # Initialize context if needed
            if user_email not in self.contexts:
                self.contexts[user_email] = ChatContext(user_email)
            
            context = self.contexts[user_email]
            context.conversation_history.append({"user": message, "timestamp": datetime.now()})
            
            # Extract intent
            intent = self._extract_intent(message)
            context.current_intent = intent
            
            # Extract entities
            entities = self._extract_entities(message)
            context.extracted_entities.update(entities)
            
            # Generate response based on intent
            response = self._generate_contextual_response(context, message)
            
            context.conversation_history.append({"assistant": response, "timestamp": datetime.now()})
            
            return {
                "intent": intent.value if intent else "unknown",
                "entities": entities,
                "response": response,
                "context": context.extracted_entities
            }
            
        except Exception as e:
            print(f"[Chatbot] Failed to process message: {e}")
            return {
                "intent": "unknown",
                "entities": {},
                "response": "I'm having trouble understanding your request. Could you please rephrase?",
                "context": {}
            }
    
    def _extract_intent(self, message: str) -> IntentType:
        """Extract user intent from message"""
        message_lower = message.lower()
        
        # Rule-based intent detection
        for intent_type, keywords in self.intent_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent_type
        
        # Use AI model if available
        if ai_models.initialized and ai_models.intent_classifier:
            try:
                labels = [intent.value for intent in IntentType]
                result = ai_models.intent_classifier(message, labels)
                if result['scores'][0] > 0.5:  # Confidence threshold
                    return IntentType(result['labels'][0])
            except Exception as e:
                print(f"[Chatbot] AI intent classification failed: {e}")
        
        return IntentType.SEARCH  # Default fallback
    
    def _extract_entities(self, message: str) -> dict:
        """Extract entities from message"""
        entities = {}
        
        # Extract price mentions
        price_patterns = [
            r'under\s*₹?(\d+)',
            r'below\s*₹?(\d+)',
            r'less\s*than\s*₹?(\d+)',
            r'₹(\d+)',
            r'(\d+)\s*rupees?'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                entities['price'] = int(matches[0])
                break
        
        # Extract categories
        products = load_products()
        categories = set(p['category'].lower() for p in products)
        message_lower = message.lower()
        
        for category in categories:
            if category in message_lower:
                entities['category'] = category
                break
        
        # Extract brand mentions
        brands = ['nike', 'boat', 'apple', 'samsung', 'sony', 'lg', 'oneplus']
        for brand in brands:
            if brand in message_lower:
                entities['brand'] = brand
                break
        
        # Extract quantities
        quantity_match = re.search(r'(\d+)\s*(?:piece|item|qty|quantity)', message_lower)
        if quantity_match:
            entities['quantity'] = int(quantity_match.group(1))
        
        return entities
    
    def _generate_contextual_response(self, context: ChatContext, message: str) -> str:
        """Generate contextual response based on intent and entities"""
        intent = context.current_intent
        entities = context.extracted_entities
        
        if intent == IntentType.SEARCH:
            if 'category' in entities:
                return f"I'll search for {entities['category']} products for you."
            else:
                return "I'll search for products matching your query."
        
        elif intent == IntentType.RECOMMENDATION:
            if 'category' in entities:
                return f"Let me recommend some great {entities['category']} products for you."
            else:
                return "I'll get personalized recommendations based on your preferences."
        
        elif intent == IntentType.COMPARISON:
            return "I can help you compare products. Please specify which products you'd like to compare."
        
        elif intent == IntentType.PRICE_INQUIRY:
            if 'price' in entities:
                return f"I'll find products under ₹{entities['price']} for you."
            else:
                return "I can help you find products within your budget. What's your price range?"
        
        elif intent == IntentType.AVAILABILITY:
            return "I'll check the availability of products for you."
        
        else:
            return "I'm here to help you find and explore products. What are you looking for?"

chatbot = IntelligentChatbot()

# ==================== ADVANCED RECOMMENDATION SYSTEM ====================

class AdvancedRecommendationSystem:
    """Multi-strategy recommendation system"""
    
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.hybrid_weights = {'content': 0.6, 'collaborative': 0.4}
    
    def get_hybrid_recommendations(self, email: str, count: int = 10) -> list[dict]:
        """Get recommendations using hybrid approach"""
        try:
            # Get content-based recommendations
            content_recs = self.content_recommender.recommend(email, count * 2)
            
            # Get collaborative filtering recommendations
            collab_recs = self.collaborative_recommender.recommend(email, count * 2)
            
            # Combine using weighted scoring
            all_products = {}
            
            # Add content-based scores
            for i, product in enumerate(content_recs):
                product_id = product['id']
                score = (count * 2 - i) * self.hybrid_weights['content']
                all_products[product_id] = {
                    'product': product,
                    'score': score
                }
            
            # Add collaborative scores
            for i, product in enumerate(collab_recs):
                product_id = product['id']
                collab_score = (count * 2 - i) * self.hybrid_weights['collaborative']
                
                if product_id in all_products:
                    all_products[product_id]['score'] += collab_score
                else:
                    all_products[product_id] = {
                        'product': product,
                        'score': collab_score
                    }
            
            # Sort by combined score
            sorted_products = sorted(
                all_products.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            return [item['product'] for item in sorted_products[:count]]
            
        except Exception as e:
            print(f"[Hybrid Recommendations] Failed: {e}")
            return get_personalized_recommendations(email, count)

class ContentBasedRecommender:
    """Content-based recommendation using product features"""
    
    def recommend(self, email: str, count: int = 10) -> list[dict]:
        """Recommend based on user's product interaction history"""
        try:
            # Get user's interaction history
            wishlist_ids = _wishlist.get(email, [])
            cart_items = _cart.get(email, [])
            cart_ids = [item.product_id for item in cart_items]
            
            interacted_ids = set(wishlist_ids + cart_ids)
            
            if not interacted_ids:
                return []
            
            # Get all products
            products = load_products()
            interacted_products = [p for p in products if p['id'] in interacted_ids]
            candidate_products = [p for p in products if p['id'] not in interacted_ids]
            
            # Calculate similarity between user profile and candidates
            recommendations = []
            
            for candidate in candidate_products:
                similarity_score = 0
                
                for interacted in interacted_products:
                    # Category similarity
                    if candidate['category'] == interacted['category']:
                        similarity_score += 0.4
                    
                    # Price similarity
                    price_diff = abs(candidate['price'] - interacted['price'])
                    price_similarity = max(0, 1 - price_diff / max(candidate['price'], interacted['price']))
                    similarity_score += price_similarity * 0.3
                    
                    # Rating similarity
                    rating_diff = abs(candidate.get('rating', 0) - interacted.get('rating', 0))
                    rating_similarity = max(0, 1 - rating_diff / 5)
                    similarity_score += rating_similarity * 0.3
                
                recommendations.append((candidate, similarity_score))
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return [product for product, score in recommendations[:count]]
            
        except Exception as e:
            print(f"[Content-Based Recommender] Failed: {e}")
            return []

class CollaborativeFilteringRecommender:
    """Collaborative filtering using user similarity"""
    
    def recommend(self, email: str, count: int = 10) -> list[dict]:
        """Recommend based on similar users' preferences"""
        try:
            # Find similar users
            similar_users = personalization_engine.find_similar_users(email, 5)
            
            if not similar_users:
                return []
            
            # Collect products liked by similar users
            recommended_products = {}
            
            for similar_user in similar_users:
                user_wishlist = _wishlist.get(similar_user, [])
                user_cart = [item.product_id for item in _cart.get(similar_user, [])]
                
                liked_products = set(user_wishlist + user_cart)
                
                for product_id in liked_products:
                    if product_id not in recommended_products:
                        recommended_products[product_id] = 0
                    recommended_products[product_id] += 1
            
            # Remove products already in user's wishlist/cart
            user_wishlist = set(_wishlist.get(email, []))
            user_cart = set(item.product_id for item in _cart.get(email, []))
            user_products = user_wishlist.union(user_cart)
            
            filtered_recommendations = {
                pid: score for pid, score in recommended_products.items()
                if pid not in user_products
            }
            
            # Sort by recommendation score
            sorted_recommendations = sorted(
                filtered_recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get product objects
            products = load_products()
            product_dict = {p['id']: p for p in products}
            
            recommended_products_list = []
            for product_id, score in sorted_recommendations[:count]:
                if product_id in product_dict:
                    recommended_products_list.append(product_dict[product_id])
            
            return recommended_products_list
            
        except Exception as e:
            print(f"[Collaborative Filtering] Failed: {e}")
            return []

advanced_recommender = AdvancedRecommendationSystem()

# ==================== ADVANCED TOOLS ====================

# class SemanticSearchEngine:
# """Advanced semantic search with multiple embedding strategies"""
# def __init__(self):
#     self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
#     self.product_embeddings = None
#     self.product_tfidf = None
#     self.query_cache = {}

#     def semantic_search(self, query: str, products: list[dict], top_k: int = 10) -> list[dict]:
#         """Advanced semantic search with multiple scoring strategies"""
#         if query in self.query_cache:
#             results = self.query_cache[query][:top_k]
#             self._display_product(results[0])  # Display the top result
#             return results

#         try:
#             # TF-IDF based search
#             query_tfidf = self.tfidf_vectorizer.transform([query])
#             tfidf_scores = cosine_similarity(query_tfidf, self.product_tfidf).flatten()

#             # Combine with other scoring strategies
#             final_scores = []
#             for i, product in enumerate(products):
#                 score = tfidf_scores[i]

#                 # Boost based on query-product matching
#                 score += self._calculate_semantic_boost(query, product)

#                 # Boost based on product popularity
#                 score += self._calculate_popularity_boost(product)

#                 final_scores.append((product, score))

#             # Sort by score and cache results
#             final_scores.sort(key=lambda x: x[1], reverse=True)
#             results = [product for product, score in final_scores]

#             self.query_cache[query] = results

#             # Display the top result
#             self._display_product(results[0])
#             return results[:top_k]

#         except Exception as e:
#             print(f"[Semantic Search] Search failed: {e}")
#             return products[:top_k]

# def _display_product(self, product: dict):
#     """Display product details in a user-friendly format"""
#     print("\n[Product Found]")
#     print(f"Name: {product.get('name', 'N/A')}")
#     print(f"Description: {product.get('description', 'N/A')}")
#     print(f"Category: {product.get('category', 'N/A')}")
#     print(f"Price: ${product.get('price', 'N/A')}")
#     print(f"Rating: {product.get('rating', 'N/A')} stars")
#     print(f"Stock: {'In stock' if product.get('inStock', True) else 'Out of stock'}")

@mcp.tool()
def intelligent_chat(email: str, message: str) -> str:
    """Process user message with advanced NLP and context awareness"""
    try:
        result = chatbot.process_message(email, message)
        
        # Execute action based on intent
        if result['intent'] == 'search':
            entities = result['entities']
            query = message
            
            # Enhance query with entities
            if 'category' in entities:
                query += f" {entities['category']}"
            if 'price' in entities:
                query += f" under {entities['price']}"
            
            search_results = smart_search(email, query)
            return f"{result['response']}\n\n{search_results}"
        
        elif result['intent'] == 'recommendation':
            recommendations = get_hybrid_recommendations(email)
            return f"{result['response']}\n\n{recommendations}"
        
        elif result['intent'] == 'comparison':
            return f"{result['response']}\n\nPlease specify the product names you'd like to compare."
        
        else:
            return result['response']
            
    except Exception as e:
        print(f"[Intelligent Chat] Failed: {e}")
        return "I'm having trouble processing your request. Could you please try again?"

@mcp.tool()
def get_hybrid_recommendations(email: str, count: int = 8) -> str:
    """Get advanced hybrid recommendations"""
    try:
        recommendations = advanced_recommender.get_hybrid_recommendations(email, count)
        
        if not recommendations:
            return "I need more information about your preferences to provide better recommendations. Try browsing some products first!"
        
        # Analyze recommendations
        categories = {}
        price_ranges = {'low': 0, 'medium': 0, 'high': 0}
        
        for product in recommendations:
            # Count categories
            category = product['category']
            categories[category] = categories.get(category, 0) + 1
            
            # Count price ranges
            price = product['price']
            if price < 2000:
                price_ranges['low'] += 1
            elif price < 10000:
                price_ranges['medium'] += 1
            else:
                price_ranges['high'] += 1
        
        # Create recommendation display
        headers = ["🎯 Hybrid Recommendations", "Category", "Price ₹", "Rating", "Match Score"]
        rows = []
        
        for i, product in enumerate(recommendations):
            # Calculate match score based on user behavior
            behavior = analyze_user_behavior(email)
            match_score = "High"
            
            if behavior.get('top_categories', {}).get(product['category']):
                match_score = "Very High"
            elif product.get('rating', 0) >= 4.5:
                match_score = "High"
            else:
                match_score = "Medium"
            
            rows.append([
                product['name'][:35] + "..." if len(product['name']) > 35 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                match_score
            ])
        
        result = "### 🤖 AI-Powered Hybrid Recommendations\n\n"
        result += tabulate(rows, headers, tablefmt="github")
        
        # Add insights
        result += "\n\n### 📊 Recommendation Insights\n"
        result += f"**Top Categories:** {', '.join(list(categories.keys())[:3])}\n"
        result += f"**Price Distribution:** {price_ranges['low']} budget-friendly, {price_ranges['medium']} mid-range, {price_ranges['high']} premium\n"
        result += f"**Recommendation Strategy:** Content-based (60%) + Collaborative filtering (40%)"
        
        return result
        
    except Exception as e:
        print(f"[Hybrid Recommendations] Failed: {e}")
        return "I'm having trouble generating recommendations right now. Please try again."

@mcp.tool()
def semantic_product_search(email: str, query: str, limit: int = 10) -> str:
    """Advanced semantic search with AI-powered understanding"""
    try:
        # Initialize semantic engine if needed
        products = load_products()
        if semantic_engine.product_tfidf is None:
            semantic_engine.create_advanced_embeddings(products)
        
        # Perform semantic search
        results = semantic_engine.semantic_search(query, products, limit)
        
        if not results:
            return "No products found matching your search criteria."
        
        # Track search
        track_search(email, query, len(results))
        
        # Create enhanced display
        headers = ["🔍 Semantic Results", "Category", "Price ₹", "Rating", "Relevance"]
        rows = []
        
        for i, product in enumerate(results):
            # Calculate relevance score
            relevance_score = "High" if i < 3 else "Medium" if i < 7 else "Low"
            
            rows.append([
                product['name'][:40] + "..." if len(product['name']) > 40 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                relevance_score
            ])
        
        result = f"### 🧠 AI-Powered Semantic Search Results for '{query}'\n\n"
        result += tabulate(rows, headers, tablefmt="github")
        
        # Add search insights
        categories = {}
        for product in results:
            cat = product['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        result += f"\n\n**Search Insights:** Found {len(results)} products across {len(categories)} categories"
        if categories:
            top_category = max(categories, key=categories.get)
            result += f", primarily in {top_category}"
        
        return result
        
    except Exception as e:
        print(f"[Semantic Search] Failed: {e}")
        return "I'm having trouble with semantic search right now. Please try again."

@mcp.tool()
def analyze_user_sentiment(email: str, feedback: str) -> str:
    """Analyze user sentiment and provide appropriate response"""
    try:
        # Initialize AI models if needed
        if not ai_models.initialized:
            ai_models.initialize_models()
        
        if not ai_models.sentiment_analyzer:
            return "Sentiment analysis is not available right now."
        
        # Analyze sentiment
        sentiment_result = ai_models.sentiment_analyzer(feedback)
        sentiment = sentiment_result[0]['label']
        confidence = sentiment_result[0]['score']
        
        # Generate appropriate response
        if sentiment == 'POSITIVE':
            response = "😊 Thank you for your positive feedback! I'm glad I could help you find what you're looking for."
        elif sentiment == 'NEGATIVE':
            response = "😔 I'm sorry to hear you're not satisfied. Let me help you find better options or resolve any issues."
        else:
            response = "🤔 I understand. Let me know how I can better assist you with your shopping needs."
        
        # Add confidence level
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        return f"{response}\n\n*Sentiment: {sentiment.lower()} (confidence: {confidence_level})*"
        
    except Exception as e:
        print(f"[Sentiment Analysis] Failed: {e}")
        return "Thank you for your feedback! I'm here to help with any questions you have."


def analyze_user_behavior(email: str) -> Dict:
    """Analyze user behavior patterns"""
    searches = _search_history.get(email, [])
    if not searches:
        return {}
    
    # Analyze search patterns
    search_terms = [s['query'].lower() for s in searches]
    category_interest = Counter()
    brand_interest = Counter()
    price_queries = []
    
    for search in searches:
        query = search['query'].lower()
        # Extract categories
        for product in load_products():
            if any(word in query for word in product['category'].lower().split()):
                category_interest[product['category']] += 1
            if 'nike' in query:
                brand_interest['Nike'] += 1
            elif 'boat' in query:
                brand_interest['boAt'] += 1
        
        # Extract price preferences
        price_match = re.search(r'under (\d+)|below (\d+)|less than (\d+)', query)
        if price_match:
            price = int(price_match.group(1) or price_match.group(2) or price_match.group(3))
            price_queries.append(price)
    
    return {
        'search_count': len(searches),
        'top_categories': dict(category_interest.most_common(3)),
        'top_brands': dict(brand_interest.most_common(3)),
        'avg_price_preference': statistics.mean(price_queries) if price_queries else None,
        'last_search': searches[-1]['timestamp'] if searches else None
    }

def get_personalized_recommendations(email: str, count: int = 5) -> List[Dict]:
    """Generate personalized recommendations based on user behavior"""
    if email in _recommendations_cache:
        cache_time = _recommendations_cache[email].get('timestamp', datetime.now())
        if datetime.now() - cache_time < timedelta(hours=1):
            return _recommendations_cache[email]['recommendations']
    
    behavior = analyze_user_behavior(email)
    products = load_products()
    wishlist_items = _wishlist.get(email, [])
    
    scored_products = []
    
    for product in products:
        if product['id'] in wishlist_items:
            continue  # Skip wishlist items
            
        score = 0
        
        # Category preference scoring
        if behavior.get('top_categories'):
            if product['category'] in behavior['top_categories']:
                score += behavior['top_categories'][product['category']] * 10
        
        # Brand preference scoring
        if behavior.get('top_brands'):
            for brand in behavior['top_brands']:
                if brand.lower() in product['name'].lower():
                    score += behavior['top_brands'][brand] * 5
        
        # Price preference scoring
        if behavior.get('avg_price_preference'):
            price_diff = abs(product['price'] - behavior['avg_price_preference'])
            score += max(0, 100 - price_diff / 100)
        
        # Rating and reviews boost
        score += product.get('rating', 0) * 5
        score += min(product.get('reviews', 0) / 1000, 10)
        
        # In-stock preference
        if product.get('inStock', True):
            score += 20
        
        # Featured products boost
        if product.get('featured', False):
            score += 15
        
        scored_products.append((product, score))
    
    # Sort by score and return top recommendations
    scored_products.sort(key=lambda x: x[1], reverse=True)
    recommendations = [product for product, score in scored_products[:count]]
    
    # Cache recommendations
    _recommendations_cache[email] = {
        'recommendations': recommendations,
        'timestamp': datetime.now()
    }
    
    return recommendations

def track_search(email: str, query: str, results_count: int = 0):
    """Track user searches for analytics"""
    _search_history[email].append({
        'query': query,
        'timestamp': datetime.now(),
        'results_count': results_count
    })
    
    # Keep only last 100 searches
    if len(_search_history[email]) > 100:
        _search_history[email] = _search_history[email][-100:]

# ---------------- Advanced Tools ---------------- #
@mcp.tool()
def get_recommendations(email: str, count: int = 5) -> str:
    """Get personalized product recommendations for the user"""
    try:
        recommendations = get_personalized_recommendations(email, count)
        
        if not recommendations:
            return "I don't have enough information about your preferences yet. Try searching for some products first!"
        
        headers = ["Recommended", "Category", "Price ₹", "Rating", "Why Recommended"]
        rows = []
        
        behavior = analyze_user_behavior(email)
        
        for product in recommendations:
            # Generate reason
            reasons = []
            if behavior.get('top_categories') and product['category'] in behavior['top_categories']:
                reasons.append(f"You like {product['category']}")
            if product.get('featured'):
                reasons.append("Popular item")
            if product.get('rating', 0) >= 4.5:
                reasons.append("High rated")
            
            reason = ", ".join(reasons) if reasons else "Based on trends"
            
            rows.append([
                product['name'][:30] + "..." if len(product['name']) > 30 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                reason
            ])
        
        return f"### 🎯 Personalized Recommendations for You\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] get_recommendations failed: {e}")
        return "I'm having trouble generating recommendations right now. Please try again."

# Fix 1: Add product validation helper function
def validate_product_exists(product_id: int) -> bool:
    """Validate if a product exists in the catalog"""
    products = load_products()
    return any(p['id'] == product_id for p in products)

def get_product_by_id(product_id: int) -> dict | None:
    """Get product by ID from catalog"""
    products = load_products()
    return next((p for p in products if p['id'] == product_id), None)

# Fix 2: Enhanced search with strict catalog matching
def search_products_strict(query: str, limit: int = 5) -> list[dict]:
    """Search products with strict catalog matching"""
    try:
        vdb = get_vectordb()
        matches = vdb.similarity_search(query, k=limit * 2)  # Get more results to filter
        
        # Filter to only include products that exist in catalog
        products = load_products()
        product_ids = {p['id'] for p in products}
        
        valid_matches = []
        for doc in matches:
            product_id = doc.metadata.get('id')
            if product_id in product_ids:
                valid_matches.append(doc.metadata)
            if len(valid_matches) >= limit:
                break
        
        return valid_matches
    except Exception as e:
        print(f"[ERROR] search_products_strict failed: {e}")
        return []

# Fix 3: Update search_products to not require email
@mcp.tool()
def search_products(query: str, email: str = None) -> str:
    """Search for products based on a query"""
    try:
        matches = search_products_strict(query, 5)
        
        if not matches:
            return "Sorry, I couldn't find any relevant products for your search."
        
        # Track the search if email provided
        if email:
            track_search(email, query, len(matches))
        
        headers = ["Name", "Category", "Price ₹", "Rating", "Reviews", "Stock"]
        rows = []
        
        for p in matches:
            rows.append([
                p.get("name", "N/A"),
                p.get("category", "N/A"),
                f"₹{p.get('price', 'N/A')}",
                p.get("rating", "N/A"),
                p.get("reviews", "N/A"),
                "✅ In Stock" if p.get("inStock", False) else "❌ Out of Stock"
            ])
        
        return "### 🛍️ Product Search Results\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] search_products failed: {e}")
        return "I'm having trouble searching for products right now. Please try again."

# Fix 4: Update smart_search with strict validation
@mcp.tool()
def smart_search(email: str = None, query: str = "", sort_by: str = "relevance") -> str:
    """Enhanced search with sorting, filtering, and personalization"""
    try:
        matches = search_products_strict(query, 10)
        
        if not matches:
            return "Sorry, I couldn't find any relevant products for your search."
        # Track the search if email provided
        if email:
            track_search(email, query, len(matches))
            behavior = analyze_user_behavior(email)
        else:
            behavior = {}
        # Score and sort products
        scored_products = []
        for product in matches:
            score = 0
            
            # Base relevance score
            score += 100
            # Personalization boost (only if email provided)
            if behavior.get('top_categories') and product['category'] in behavior['top_categories']:
                score += 20
            # Rating boost
            score += product.get('rating', 0) * 5
            # Stock availability boost
            if product.get('inStock', True):
                score += 10
            # Featured product boost
            if product.get('featured', False):
                score += 15
            scored_products.append((product, score))
        
        # Sort based on user preference
        if sort_by == "price_low":
            scored_products.sort(key=lambda x: x[0]['price'])
        elif sort_by == "price_high":
            scored_products.sort(key=lambda x: x[0]['price'], reverse=True)
        elif sort_by == "rating":
            scored_products.sort(key=lambda x: x[0].get('rating', 0), reverse=True)
        elif sort_by == "reviews":
            scored_products.sort(key=lambda x: x[0].get('reviews', 0), reverse=True)
        else:  # relevance
            scored_products.sort(key=lambda x: x[1], reverse=True)
        
        headers = ["Product", "Category", "Price ₹", "Rating", "Stock", "Reviews"]
        rows = []
        
        for product, score in scored_products[:8]:  # Top 8 results
            rows.append([
                product['name'][:35] + "..." if len(product['name']) > 35 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                "✅" if product.get('inStock', True) else "❌",
                product.get('reviews', 'N/A')
            ])
        
        result = f"### 🔍 Smart Search Results for '{query}'\n"
        result += f"*Sorted by: {sort_by.replace('_', ' ').title()}*\n\n"
        result += tabulate(rows, headers, tablefmt="github")
        
        return result
    
    except Exception as e:
        print(f"[ERROR] smart_search failed: {e}")
        return "I'm having trouble with smart search right now. Please try again."

# Fix 5: Update add_to_wishlist with strict validation
@mcp.tool()
def add_to_wishlist(email: str, product_name: str) -> str:
    """Add a product to user's wishlist"""
    try:
        matches = search_products_strict(product_name, 1)
        
        if not matches:
            return f"Sorry, I couldn't find a product named '{product_name}' in our catalog."
        
        product = matches[0]
        product_id = product.get('id')
        
        # Double-check product exists
        if not validate_product_exists(product_id):
            return f"Sorry, '{product_name}' is not available in our catalog."
        
        if product_id in _wishlist[email]:
            return f"**{product['name']}** is already in your wishlist!"
        
        _wishlist[email].append(product_id)
        
        # Save to database if available
        if db is not None:
            try:
                db["Users"].update_one(
                    {"email": email},
                    {"$addToSet": {"wishlist": product_id}},
                    upsert=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to save wishlist to DB: {e}")
        
        return f"✅ Added **{product['name']}** to your wishlist!"
    
    except Exception as e:
        print(f"[ERROR] add_to_wishlist failed: {e}")
        return "I'm having trouble adding to your wishlist right now. Please try again."

# Fix 6: Update add_to_cart with strict validation
@mcp.tool()
def add_to_cart(email: str, product_name: str, quantity: int = 1) -> str:
    """Add a product to user's cart"""
    try:
        if quantity <= 0:
            return "Please specify a valid quantity (greater than 0)."
        
        matches = search_products_strict(product_name, 1)
        
        if not matches:
            return f"Sorry, I couldn't find a product named '{product_name}' in our catalog."
        
        product = matches[0]
        product_id = product.get('id')
        
        # Double-check product exists
        if not validate_product_exists(product_id):
            return f"Sorry, '{product_name}' is not available in our catalog."
        
        if not product.get('inStock', True):
            return f"Sorry, **{product['name']}** is currently out of stock."
        
        # Check if product already in cart
        for item in _cart[email]:
            if item.product_id == product_id:
                item.quantity += quantity
                return f"✅ Updated **{product['name']}** quantity to {item.quantity} in your cart!"
        
        # Add new item to cart
        cart_item = CartItem(
            product_id=product_id,
            quantity=quantity,
            added_at=datetime.now()
        )
        _cart[email].append(cart_item)
        
        return f"✅ Added **{product['name']}** (×{quantity}) to your cart!"
    
    except Exception as e:
        print(f"[ERROR] add_to_cart failed: {e}")
        return "I'm having trouble adding to your cart right now. Please try again."

# Fix 7: Update get_recommendations with strict validation
@mcp.tool()
def get_recommendations(email: str = None, count: int = 5) -> str:
    """Get personalized product recommendations for the user"""
    try:
        if email:
            recommendations = get_personalized_recommendations(email, count)
        else:
            # If no email, return general trending products
            products = load_products()
            recommendations = sorted(products, key=lambda x: x.get('rating', 0), reverse=True)[:count]
        
        if not recommendations:
            return "I don't have enough information about your preferences yet. Try searching for some products first!"
        
        headers = ["Recommended", "Category", "Price ₹", "Rating", "Why Recommended"]
        rows = []
        
        behavior = analyze_user_behavior(email) if email else {}
        
        for product in recommendations:
            # Ensure product exists in catalog
            if not validate_product_exists(product.get('id')):
                continue
                
            # Generate reason
            reasons = []
            if behavior.get('top_categories') and product['category'] in behavior['top_categories']:
                reasons.append(f"You like {product['category']}")
            if product.get('featured'):
                reasons.append("Popular item")
            if product.get('rating', 0) >= 4.5:
                reasons.append("High rated")
            
            reason = ", ".join(reasons) if reasons else "Based on trends"
            
            rows.append([
                product['name'][:30] + "..." if len(product['name']) > 30 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                reason
            ])
        
        if not rows:
            return "No recommendations available at the moment."
        
        return f"### 🎯 Personalized Recommendations for You\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] get_recommendations failed: {e}")
        return "I'm having trouble generating recommendations right now. Please try again."

# Fix 8: Update get_personalized_recommendations with strict validation
def get_personalized_recommendations(email: str, count: int = 5) -> List[Dict]:
    """Generate personalized recommendations based on user behavior"""
    if email in _recommendations_cache:
        cache_time = _recommendations_cache[email].get('timestamp', datetime.now())
        if datetime.now() - cache_time < timedelta(hours=1):
            return _recommendations_cache[email]['recommendations']
    
    behavior = analyze_user_behavior(email)
    products = load_products()
    wishlist_items = _wishlist.get(email, [])
    
    # Filter to only valid products
    valid_products = [p for p in products if validate_product_exists(p.get('id'))]
    
    scored_products = []
    
    for product in valid_products:
        if product['id'] in wishlist_items:
            continue  # Skip wishlist items
            
        score = 0
        
        # Category preference scoring
        if behavior.get('top_categories'):
            if product['category'] in behavior['top_categories']:
                score += behavior['top_categories'][product['category']] * 10
        
        # Brand preference scoring
        if behavior.get('top_brands'):
            for brand in behavior['top_brands']:
                if brand.lower() in product['name'].lower():
                    score += behavior['top_brands'][brand] * 5
        
        # Price preference scoring
        if behavior.get('avg_price_preference'):
            price_diff = abs(product['price'] - behavior['avg_price_preference'])
            score += max(0, 100 - price_diff / 100)
        
        # Rating and reviews boost
        score += product.get('rating', 0) * 5
        score += min(product.get('reviews', 0) / 1000, 10)
        
        # In-stock preference
        if product.get('inStock', True):
            score += 20
        
        # Featured products boost
        if product.get('featured', False):
            score += 15
        
        scored_products.append((product, score))
    
    # Sort by score and return top recommendations
    scored_products.sort(key=lambda x: x[1], reverse=True)
    recommendations = [product for product, score in scored_products[:count]]
    
    # Cache recommendations
    _recommendations_cache[email] = {
        'recommendations': recommendations,
        'timestamp': datetime.now()
    }
    
    return recommendations

# Fix 9: Update compare_products with strict validation
@mcp.tool()
def compare_products(product_names: list[str]) -> str:
    """Compare multiple products side by side"""
    try:
        if len(product_names) < 2:
            return "Please provide at least 2 product names to compare."
        
        chosen_products = []
        
        for name in product_names:
            matches = search_products_strict(name, 1)
            if matches:
                chosen_products.append(matches[0])
        
        if len(chosen_products) < 2:
            return "I couldn't find enough products to compare in our catalog. Please check the product names."
        
        headers = ["Product", "Price ₹", "Rating", "Reviews", "Stock", "Category"]
        rows = []
        
        for p in chosen_products:
            rows.append([
                p.get("name", "N/A"),
                f"₹{p.get('price', 'N/A')}",
                p.get("rating", "N/A"),
                p.get("reviews", "N/A"),
                "✅" if p.get("inStock", False) else "❌",
                p.get("category", "N/A")
            ])
        
        return "### ⚖️ Product Comparison\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] compare_products failed: {e}")
        return "I'm having trouble comparing products right now. Please try again."

# Fix 10: Update check_availability with strict validation
@mcp.tool()
def check_availability(product_name: str) -> str:
    """Check if a specific product is available"""
    try:
        matches = search_products_strict(product_name, 1)
        
        if not matches:
            return f"Sorry, I couldn't find a product named '{product_name}' in our catalog."
        
        p = matches[0]
        product_name = p.get("name", "Unknown Product")
        price = p.get("price", "N/A")
        in_stock = p.get("inStock", False)
        
        if in_stock:
            return f"✅ **{product_name}** is available for ₹{price}!"
        else:
            return f"❌ Sorry, **{product_name}** is currently out of stock."
    
    except Exception as e:
        print(f"[ERROR] check_availability failed: {e}")
        return "I'm having trouble checking product availability right now. Please try again."

# Fix 11: Update get_product_details with strict validation
@mcp.tool()
def get_product_details(product_name: str) -> str:
    """Get detailed information about a specific product"""
    try:
        matches = search_products_strict(product_name, 1)
        
        if not matches:
            return f"Sorry, I couldn't find detailed information about '{product_name}' in our catalog."
        
        p = matches[0]
        
        details = f"""
### 📱 {p.get('name', 'Unknown Product')}

**Description**: {p.get('description', 'No description available')}
**Category**: {p.get('category', 'N/A')}
**Price**: ₹{p.get('price', 'N/A')}
**Rating**: {p.get('rating', 'N/A')} ⭐
**Reviews**: {p.get('reviews', 'N/A')} customer reviews
**Availability**: {'✅ In Stock' if p.get('inStock', False) else '❌ Out of Stock'}
"""
        
        return details.strip()
    
    except Exception as e:
        print(f"[ERROR] get_product_details failed: {e}")
        return "I'm having trouble getting product details right now. Please try again."

# Fix 12: Update get_price_alerts with strict validation
@mcp.tool()
def get_price_alerts(email: str, product_name: str, target_price: float) -> str:
    """Set price alert for a product"""
    try:
        matches = search_products_strict(product_name, 1)
        
        if not matches:
            return f"Sorry, I couldn't find a product named '{product_name}' in our catalog."
        
        product = matches[0]
        current_price = product.get('price', 0)
        
        if target_price >= current_price:
            return f"**{product['name']}** is already priced at ₹{current_price}, which is below your target of ₹{target_price}!"
        
        return f"✅ Price alert set for **{product['name']}**!\n\n📍 Current Price: ₹{current_price}\n🎯 Target Price: ₹{target_price}\n\nI'll notify you when the price drops to ₹{target_price} or below."
    
    except Exception as e:
        print(f"[ERROR] get_price_alerts failed: {e}")
        return "I'm having trouble setting up price alerts right now. Please try again."


@mcp.tool()
def view_wishlist(email: str) -> str:
    """View user's wishlist"""
    try:
        wishlist_ids = _wishlist.get(email, [])
        
        if not wishlist_ids:
            return "Your wishlist is empty! Start adding products you like."
        
        products = load_products()
        wishlist_products = [p for p in products if p['id'] in wishlist_ids]
        
        if not wishlist_products:
            return "Your wishlist is empty! Start adding products you like."
        
        headers = ["Product", "Category", "Price ₹", "Rating", "Stock"]
        rows = []
        
        for product in wishlist_products:
            rows.append([
                product['name'][:40] + "..." if len(product['name']) > 40 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                "✅" if product.get('inStock', True) else "❌"
            ])
        
        return f"### 💝 Your Wishlist ({len(wishlist_products)} items)\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] view_wishlist failed: {e}")
        return "I'm having trouble accessing your wishlist right now. Please try again."

@mcp.tool()
def remove_from_wishlist(email: str, product_name: str) -> str:
    """Remove a product from user's wishlist"""
    try:
        vdb = get_vectordb()
        hits = vdb.similarity_search(product_name, k=1)
        
        if not hits:
            return f"Sorry, I couldn't find a product named '{product_name}'."
        
        product = hits[0].metadata
        product_id = product.get('id')
        
        if product_id not in _wishlist[email]:
            return f"**{product['name']}** is not in your wishlist."
        
        _wishlist[email].remove(product_id)
        
        # Remove from database if available
        if db is not None:
            try:
                db["Users"].update_one(
                    {"email": email},
                    {"$pull": {"wishlist": product_id}}
                )
            except Exception as e:
                print(f"[ERROR] Failed to remove from wishlist in DB: {e}")
        
        return f"✅ Removed **{product['name']}** from your wishlist!"
    
    except Exception as e:
        print(f"[ERROR] remove_from_wishlist failed: {e}")
        return "I'm having trouble removing from your wishlist right now. Please try again."



@mcp.tool()
def view_cart(email: str) -> str:
    """View user's shopping cart"""
    try:
        cart_items = _cart.get(email, [])
        
        if not cart_items:
            return "Your cart is empty! Start adding products to buy."
        
        products = load_products()
        headers = ["Product", "Price ₹", "Qty", "Total ₹", "Added"]
        rows = []
        total_amount = 0
        
        for item in cart_items:
            product = next((p for p in products if p['id'] == item.product_id), None)
            if product:
                item_total = product['price'] * item.quantity
                total_amount += item_total
                
                rows.append([
                    product['name'][:30] + "..." if len(product['name']) > 30 else product['name'],
                    f"₹{product['price']}",
                    item.quantity,
                    f"₹{item_total}",
                    item.added_at.strftime("%m/%d")
                ])
        
        cart_summary = f"### 🛒 Your Cart ({len(cart_items)} items)\n\n"
        cart_summary += tabulate(rows, headers, tablefmt="github")
        cart_summary += f"\n\n**Total Amount: ₹{total_amount}**"
        
        return cart_summary
    
    except Exception as e:
        print(f"[ERROR] view_cart failed: {e}")
        return "I'm having trouble accessing your cart right now. Please try again."



@mcp.tool()
def search_products(query: str, email: str = "anonymous") -> str:
    """Search for products based on a query (enhanced version)"""
    try:
        vdb = get_vectordb()
        matches = vdb.similarity_search(query, k=5)
        
        if not matches:
            return "Sorry, I couldn't find any relevant products for your search."
        
        # Track the search if email provided
        if email != "anonymous":
            track_search(email, query, len(matches))
        
        headers = ["Name", "Category", "Price ₹", "Rating", "Reviews", "Stock"]
        rows = []
        
        for doc in matches:
            p = doc.metadata
            rows.append([
                p.get("name", "N/A"),
                p.get("category", "N/A"),
                f"₹{p.get('price', 'N/A')}",
                p.get("rating", "N/A"),
                p.get("reviews", "N/A"),
                "✅ In Stock" if p.get("inStock", False) else "❌ Out of Stock"
            ])
        
        return "### 🛍️ Product Search Results\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] search_products failed: {e}")
        return "I'm having trouble searching for products right now. Please try again."

# ---------------- Session Management ---------------- #
def start_user_session(email: str):
    """Start a new user session"""
    _user_sessions[email] = {
        'start_time': datetime.now(),
        'last_activity': datetime.now(),
        'page_views': 0,
        'searches': 0,
        'cart_actions': 0
    }

def update_user_activity(email: str, activity_type: str):
    """Update user activity in session"""
    if email not in _user_sessions:
        start_user_session(email)
    
    session = _user_sessions[email]
    session['last_activity'] = datetime.now()
    
    if activity_type == 'search':
        session['searches'] += 1
    elif activity_type == 'cart':
        session['cart_actions'] += 1
    elif activity_type == 'view':
        session['page_views'] += 1


@mcp.tool()
def get_trending_products(category: str = "", limit: int = 10) -> str:
    """Get truly trending products based on actual user interactions"""
    try:
        products = load_products()
        
        # Count product appearances across all wishlists and carts
        product_popularity = Counter()
        for email, items in _wishlist.items():
            for product_id in items:
                product_popularity[product_id] += 1
                
        for email, items in _cart.items():
            for item in items:
                product_popularity[item.product_id] += 1
        
        # Filter by category if specified
        if category:
            products = [p for p in products if category.lower() in p['category'].lower()]
        
        # Score products
        scored_products = []
        for product in products:
            if not product.get('inStock', True):
                continue
                
            score = 0
            # Popularity from user interactions (50%)
            score += product_popularity.get(product['id'], 0) * 50
            # Rating (30%)
            score += product.get('rating', 0) * 30
            # Reviews (20%) - normalized
            score += min(product.get('reviews', 0) / 100, 20)
            
            scored_products.append((product, score))
        
        # Sort by score and limit results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        trending_products = [p for p, score in scored_products[:limit]]
        
        headers = ["🔥 Trending", "Category", "Price ₹", "Rating", "Popularity"]
        rows = []
        
        for product in trending_products:
            rows.append([
                product['name'][:40] + "..." if len(product['name']) > 40 else product['name'],
                product['category'],
                f"₹{product['price']}",
                f"{product.get('rating', 'N/A')} ⭐",
                f"{product_popularity.get(product['id'], 0)} adds"
            ])
        
        title = "### 🔥 Genuinely Trending Products"
        if category:
            title += f" in {category}"
        
        return title + "\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] get_trending_products failed: {e}")
        return "I'm having trouble getting trending products right now."

# ---------------- Load user data from database on startup ---------------- #
def load_user_data():
    """Load user data from database on startup"""
    if db is None:
        return
    
    try:
        users = db["Users"].find({})
        for user in users:
            email = user.get('email')
            if email:
                _current_user_cache[email] = user
                if 'wishlist' in user:
                    _wishlist[email] = user['wishlist']
                if 'search_history' in user:
                    _search_history[email] = user['search_history']
    except Exception as e:
        print(f"[ERROR] Failed to load user data: {e}")

def docs_from_products(products: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=(
                f"{p['name']}. {p.get('description','')}. "
                f"Category: {p['category']}. Price: ₹{p['price']}. "
                f"In‑stock: {p.get('inStock', True)}. "
                f"Rating: {p.get('rating','N/A')}. Reviews: {p.get('reviews','N/A')}."
            ),
            metadata=p,
        )
        for p in products
    ]

def get_vectordb():
    global _vectordb
    if _vectordb is None:
        try:
            emb = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            _vectordb = Chroma.from_documents(
                docs_from_products(load_products()),
                embedding=emb,
                persist_directory="./chroma_products"
            )
            print("[VectorDB] Initialized successfully")
        except Exception as e:
            print(f"[VectorDB] Initialization failed: {e}")
            raise
    return _vectordb

# ---------------- QA Chain ---------------- #
def qa_chain(vdb):
    prompt = PromptTemplate.from_template(
        """
You are a helpful ecommerce assistant. **Answer ONLY from the context**.
If the answer is not in context say:
"I'm sorry, I don't see that information yet."

Context:
---------
{context}

Question: {question}
---------
Answer (markdown):
"""
    )
    return RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
        retriever=vdb.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

# ---------------- User Management Helper ---------------- #
def get_user_by_email(email: str) -> dict | None:
    """Get user from cache or database"""
    if email in _current_user_cache:
        return _current_user_cache[email]
    
    if db is None:
        print("[WARNING] Database not available, using in-memory storage")
        return None
    
    try:
        user = db["Users"].find_one({"email": email})
        if user:
            _current_user_cache[email] = user
        return user
    except Exception as e:
        print(f"[ERROR] Database query failed: {e}")
        return None

def create_user(email: str) -> dict:
    """Create a new user"""
    user_doc = {"email": email}
    
    if db is not None:
        try:
            db["Users"].insert_one(user_doc.copy())
            print(f"[DB] Created user: {email}")
        except Exception as e:
            print(f"[ERROR] Failed to create user in DB: {e}")
    
    _current_user_cache[email] = user_doc
    return user_doc

def update_user_name(email: str, name: str) -> bool:
    """Update user name"""
    if db is not None:
        try:
            result = db["Users"].update_one(
                {"email": email}, 
                {"$set": {"name": name}}
            )
            if result.modified_count > 0:
                print(f"[DB] Updated name for {email}: {name}")
            else:
                print(f"[DB] No document updated for {email}")
        except Exception as e:
            print(f"[ERROR] Failed to update user name in DB: {e}")
            return False
    
    # Update cache
    if email in _current_user_cache:
        _current_user_cache[email]["name"] = name
    
    return True

# ---------------- Onboarding Tools ---------------- #
@mcp.tool()
def connect_user(email: str) -> str:
    """Connect a user and check if they need to provide their name"""
    try:
        user = get_user_by_email(email)
        
        if user is None:
            # Create new user
            user = create_user(email)
        
        # Check if user has a name
        if "name" not in user or not user.get("name"):
            return (
                f"👋 Hi! I see your email is **{email}**, "
                "but I don't know your name yet. What should I call you?"
            )
        
        return f"Welcome back, **{user['name']}**! How can I help you today?"
    
    except Exception as e:
        print(f"[ERROR] connect_user failed: {e}")
        return f"Hi there! I'm having trouble accessing your information right now, but I can still help you with product questions."

@mcp.tool()
def set_user_name(name: str, email: str) -> str:
    """Set the user's name"""
    try:
        if not name or not name.strip():
            return "Please provide a valid name."
        
        name = name.strip()
        
        # Validate name (basic check)
        if len(name) > 50:
            return "Name is too long. Please provide a shorter name."
        
        success = update_user_name(email, name)
        
        if success:
            return f"Thanks, **{name}**! I've saved your name. How can I help you today?"
        else:
            return f"Thanks, **{name}**! I'll remember that for this session."
    
    except Exception as e:
        print(f"[ERROR] set_user_name failed: {e}")
        return "I had trouble saving your name, but I can still help you with product questions."


@mcp.tool()
def filter_products(category: str = "", min_price: float = 0, max_price: float = 1000000, in_stock_only: bool = False) -> str:
    """Filter products by category, price range, and stock status"""
    try:
        vdb = get_vectordb()
        
        # Build search query
        query_parts = []
        if category:
            query_parts.append(f"{category} products")
        query_parts.append(f"price between {min_price} and {max_price}")
        if in_stock_only:
            query_parts.append("in stock")
        
        query = " ".join(query_parts)
        hits = vdb.similarity_search(query, k=20)
        
        # Filter results
        filtered_results = []
        for hit in hits:
            p = hit.metadata
            price = p.get("price", 0)
            in_stock = p.get("inStock", True)
            
            # Apply filters
            if price < min_price or price > max_price:
                continue
            if in_stock_only and not in_stock:
                continue
            if category and category.lower() not in p.get("category", "").lower():
                continue
            
            filtered_results.append(p)
        
        if not filtered_results:
            return "No products match your filter criteria."
        
        headers = ["Name", "Category", "Price ₹", "Rating", "Stock"]
        rows = []
        
        for p in filtered_results[:10]:  # Limit to top 10
            rows.append([
                p.get("name", "N/A"),
                p.get("category", "N/A"),
                f"₹{p.get('price', 'N/A')}",
                p.get("rating", "N/A"),
                "✅" if p.get("inStock", False) else "❌"
            ])
        
        return f"### 🔍 Filtered Products ({len(filtered_results)} found)\n\n" + tabulate(rows, headers, tablefmt="github")
    
    except Exception as e:
        print(f"[ERROR] filter_products failed: {e}")
        return "I'm having trouble filtering products right now. Please try again."

    except Exception as e:
        print(f"[ERROR] compare_products failed: {e}")
        return "I'm having trouble comparing products right now. Please try again."

@mcp.tool()
def get_policy_info(topic: str) -> str:
    """Get information about store policies"""
    policies = {
        "return": "📦 **Return Policy**: You can return items within 30 days of purchase in original condition.",
        "refund": "💰 **Refund Policy**: Refunds are processed within 5-7 business days after we receive your return.",
        "shipping": "🚚 **Shipping**: Standard shipping takes 3-5 days, express shipping takes 1-2 days.",
        "warranty": "🛡️ **Warranty**: Most electronics come with a 1-year manufacturer warranty.",
        "exchange": "🔄 **Exchange**: Items can be exchanged within 15 days for size or color changes.",
        "cancellation": "❌ **Cancellation**: Orders can be cancelled within 24 hours of placing them."
    }
    
    topic_lower = topic.lower()
    for key, value in policies.items():
        if key in topic_lower:
            return value
    
    return "I don't have specific information about that policy. Please contact customer support for more details."


@mcp.tool()
def analyze_user_patterns(email: str) -> str:
    """Analyze user's shopping patterns from cart/wishlist additions"""
    try:
        cart_items = _cart.get(email, [])
        wishlist_items = _wishlist.get(email, [])
        
        if not cart_items and not wishlist_items:
            return "No shopping patterns detected yet. Start adding items to your cart or wishlist."
        
        products = load_products()
        product_dict = {p['id']: p for p in products}
        
        # Analyze categories
        category_counter = Counter()
        for item in cart_items + wishlist_items:
            product = product_dict.get(item.product_id)
            if product:
                category_counter[product['category']] += 1
        
        # Analyze price ranges
        prices = []
        for item in cart_items + wishlist_items:
            product = product_dict.get(item.product_id)
            if product:
                prices.append(product['price'])
        
        avg_price = statistics.mean(prices) if prices else 0
        
        # Generate insights
        insights = "### 🕵️‍♂️ Your Shopping Patterns\n\n"
        insights += f"**Favorite Categories**: {', '.join([cat for cat, count in category_counter.most_common(3)])}\n"
        insights += f"**Average Price Point**: ₹{avg_price:.2f}\n"
        insights += f"**Total Items Added**: {len(cart_items) + len(wishlist_items)}\n"
        
        return insights
        
    except Exception as e:
        print(f"[ERROR] analyze_user_patterns failed: {e}")
        return "I'm having trouble analyzing your shopping patterns right now."


# ---------------- Launch ---------------- #
if __name__ == "__main__":
    print("[ProductAgent] Starting MCP tool server...")
    try:
        # Initialize vector database
        get_vectordb()
        print("[ProductAgent] Vector database initialized")
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"[ERROR] Failed to start ProductAgent: {e}")
        raise