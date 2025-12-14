import json
import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class JSONParser:
    """Robust JSON parser that handles malformed JSON files"""
    
    @staticmethod
    def clean_json_string(content: str) -> str:
        """Clean common JSON formatting issues"""
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        content = content.replace('\\"', '"').replace('\\/', '/')
        return content.strip()
    
    @staticmethod
    def extract_json_objects(content: str) -> List[Dict]:
        """Extract JSON objects from content"""
        objects = []
        depth = 0
        start_idx = None
        
        for i, char in enumerate(content):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    try:
                        obj_str = content[start_idx:i+1]
                        obj = json.loads(obj_str)
                        objects.append(obj)
                    except:
                        pass
                    start_idx = None
        
        return objects
    
    @classmethod
    def load_json(cls, filepath: str) -> Optional[Dict]:
        """Load JSON with multiple fallback strategies"""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Standard JSON parsing failed, trying cleanup...")
            
            try:
                cleaned = cls.clean_json_string(content)
                return json.loads(cleaned, strict=False)
            except json.JSONDecodeError:
                logger.warning("Cleaned JSON parsing failed, trying object extraction...")
            
            objects = cls.extract_json_objects(content)
            if objects:
                if len(objects) == 1:
                    return objects[0]
                return {"data": objects}
            
            try:
                content = content.encode('utf-8').decode('utf-8-sig')
                cleaned = cls.clean_json_string(content)
                return json.loads(cleaned, strict=False)
            except:
                pass
            
            logger.error(f"All JSON parsing strategies failed for {filepath}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None


class ConversationExtractor:
    """Extract conversations from various chat data formats"""
    
    @staticmethod
    def extract_conversations(chat_data: Dict) -> List[Dict]:
        """Extract user-AI conversation pairs from chat data"""
        conversations = []
        
        turns = None
        if 'conversation_turns' in chat_data:
            turns = chat_data['conversation_turns']
        elif 'data' in chat_data and 'conversation_turns' in chat_data['data']:
            turns = chat_data['data']['conversation_turns']
        elif 'turns' in chat_data:
            turns = chat_data['turns']
        elif 'messages' in chat_data:
            turns = chat_data['messages']
        
        if not turns:
            logger.warning("No conversation turns found in chat data")
            return []
        
        for i, turn in enumerate(turns):
            role = turn.get('role', '').lower()
            
            if any(keyword in role for keyword in ['ai', 'assistant', 'chatbot', 'bot']):
                user_turn = None
                for j in range(i-1, -1, -1):
                    prev_role = turns[j].get('role', '').lower()
                    if any(keyword in prev_role for keyword in ['user', 'human', 'customer']):
                        user_turn = turns[j]
                        break
                
                if user_turn:
                    conversations.append({
                        'turn': turn.get('turn', i+1),
                        'user_query': user_turn.get('message', ''),
                        'ai_response': turn.get('message', ''),
                        'timestamp': turn.get('created_at', ''),
                        'sender_id': turn.get('sender_id', '')
                    })
        
        logger.info(f"Extracted {len(conversations)} conversation pairs")
        return conversations


class ContextExtractor:
    """Extract and optimize context from vector database results"""
    
    @staticmethod
    def extract_context(vector_data: Dict, max_contexts: int = 5, max_chars: int = 2000) -> str:
        """
        Extract text context from vector data with optimization
        CRITICAL: Limit context size to prevent timeouts
        """
        context_texts = []
        
        vectors = None
        if isinstance(vector_data, list):
            vectors = vector_data
        elif 'data' in vector_data:
            if 'vector_data' in vector_data['data']:
                vectors = vector_data['data']['vector_data']
            elif isinstance(vector_data['data'], list):
                vectors = vector_data['data']
        elif 'vector_data' in vector_data:
            vectors = vector_data['vector_data']
        elif 'results' in vector_data:
            vectors = vector_data['results']
        
        if not vectors:
            logger.warning("No vector data found")
            return ""
        
        total_chars = 0
        for v in vectors[:max_contexts]:
            if isinstance(v, dict):
                text = v.get('text', v.get('content', v.get('document', '')))
                if text and isinstance(text, str):
                    text = text.strip()
                    # Truncate individual chunks if too long
                    if len(text) > 500:
                        text = text[:500] + "..."
                    
                    if total_chars + len(text) <= max_chars:
                        context_texts.append(text)
                        total_chars += len(text)
                    else:
                        # Add partial text to reach limit
                        remaining = max_chars - total_chars
                        if remaining > 100:
                            context_texts.append(text[:remaining] + "...")
                        break
        
        combined_context = "\n\n".join(context_texts)
        logger.info(f"Extracted {len(context_texts)} context chunks ({len(combined_context)} chars)")
        return combined_context


class LLMEvaluator:
    """Main evaluator class using Ollama with optimizations"""
    
    def __init__(self, model: str = None, ollama_url: str = None):
        self.ollama_url = ollama_url or os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.model = model or os.getenv('OLLAMA_MODEL', 'mistral:7b')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '180'))  # Increased to 180s
        self.max_tokens = int(os.getenv('MAX_TOKENS', '300'))  # Limit output tokens
        
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"✓ Connected to Ollama at {self.ollama_url}")
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to Ollama at {self.ollama_url}")
            logger.error("Please ensure Ollama is running: ollama serve")
            raise
    
    def call_ollama(self, prompt: str, use_simple: bool = False) -> Dict[str, Any]:
        """Call Ollama API with simplified mode for speed"""
        start_time = time.time()
        
        try:
            # Reduce timeout for faster failure detection
            actual_timeout = 60 if use_simple else self.timeout
            
            logger.info(f"    Calling Ollama {'(simple mode)' if use_simple else ''}...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 150 if use_simple else self.max_tokens,
                        "num_ctx": 2048 if use_simple else 4096,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_thread": 4  # Limit threads
                    }
                },
                timeout=actual_timeout
            )
            
            latency = time.time() - start_time
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            content = result.get('response', '').strip()
            
            logger.info(f"    ✓ Response received ({len(content)} chars, {latency:.2f}s)")
            
            return {
                'content': content,
                'latency_ms': round(latency * 1000, 2),
                'input_tokens': result.get('prompt_eval_count', 0),
                'output_tokens': result.get('eval_count', 0)
            }
            
        except requests.exceptions.Timeout:
            logger.warning(f"    ⚠ Timed out after {actual_timeout}s")
            raise Exception(f"Request timed out after {actual_timeout}s")
        except Exception as e:
            logger.error(f"    ✗ Ollama call failed: {str(e)}")
            raise
    
    def parse_evaluation_response(self, response: str) -> Dict:
        """Parse LLM evaluation response with fallbacks"""
        response = re.sub(r'```json\s*|\s*```', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        scores = {}
        patterns = {
            'relevance_score': r'relevance[_\s]*score["\s:]+(\d+)',
            'completeness_score': r'completeness[_\s]*score["\s:]+(\d+)',
            'hallucination_score': r'hallucination[_\s]*score["\s:]+(\d+)',
            'factual_accuracy_score': r'factual[_\s]*accuracy[_\s]*score["\s:]+(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[key] = int(match.group(1))
        
        if scores:
            return {
                **scores,
                'issues': [],
                'missing_info': [],
                'hallucinations': []
            }
        
        return {
            'relevance_score': 0,
            'completeness_score': 0,
            'hallucination_score': 0,
            'factual_accuracy_score': 0,
            'issues': ['Failed to parse evaluation'],
            'missing_info': [],
            'hallucinations': [],
            'raw_response': response[:500]
        }
    
    def create_simple_prompt(self, user_query: str, ai_response: str, context: str) -> str:
        """Create ultra-simple prompt for fast evaluation"""
        return f"""Rate 0-10:
Q: {user_query[:200]}
A: {ai_response[:300]}
Context: {context[:400]}

JSON only:
{{"relevance_score":0-10,"completeness_score":0-10,"hallucination_score":0-10,"factual_accuracy_score":0-10}}"""

    def evaluate_response(self, user_query: str, ai_response: str, context: str) -> Dict:
        """Evaluate a single AI response with multi-tier fallback strategy"""
        
        # Tier 1: Try with moderate truncation (60s timeout)
        user_query_truncated = user_query[:300]
        ai_response_truncated = ai_response[:500]
        context_truncated = context[:800]
        
        prompt = f"""Evaluate AI response. Return JSON only.

USER: {user_query_truncated}
AI: {ai_response_truncated}
CONTEXT: {context_truncated}

{{"relevance_score":0-10,"completeness_score":0-10,"hallucination_score":0-10,"factual_accuracy_score":0-10,"issues":[],"missing_info":[],"hallucinations":[]}}"""

        try:
            result = self.call_ollama(prompt, use_simple=True)
            eval_data = self.parse_evaluation_response(result['content'])
            
            eval_data['latency_ms'] = result['latency_ms']
            eval_data['input_tokens'] = result['input_tokens']
            eval_data['output_tokens'] = result['output_tokens']
            eval_data['cost_usd'] = 0.0
            eval_data['evaluation_mode'] = 'standard'
            
            return eval_data
            
        except Exception as e:
            logger.warning(f"    Standard evaluation failed: {str(e)}")
            logger.info("    Falling back to ultra-fast mode...")
        
        # Tier 2: Ultra-simple evaluation (30s timeout, minimal prompt)
        try:
            simple_prompt = self.create_simple_prompt(
                user_query, ai_response, context
            )
            result = self.call_ollama(simple_prompt, use_simple=True)
            
            # Parse simple response
            eval_data = self.parse_evaluation_response(result['content'])
            if not eval_data or eval_data.get('relevance_score', 0) == 0:
                # Try to extract just numbers
                import re
                numbers = re.findall(r'\d+', result['content'])
                if len(numbers) >= 4:
                    eval_data = {
                        'relevance_score': min(int(numbers[0]), 10),
                        'completeness_score': min(int(numbers[1]), 10),
                        'hallucination_score': min(int(numbers[2]), 10),
                        'factual_accuracy_score': min(int(numbers[3]), 10),
                        'issues': [],
                        'missing_info': [],
                        'hallucinations': []
                    }
            
            eval_data['latency_ms'] = result['latency_ms']
            eval_data['input_tokens'] = result['input_tokens']
            eval_data['output_tokens'] = result['output_tokens']
            eval_data['cost_usd'] = 0.0
            eval_data['evaluation_mode'] = 'simple'
            
            return eval_data
            
        except Exception as e:
            logger.warning(f"    Simple evaluation failed: {str(e)}")
            logger.info("    Using heuristic-based evaluation...")
        
        # Tier 3: Heuristic-based evaluation (no LLM, instant)
        return self.heuristic_evaluation(user_query, ai_response, context)
    
    def heuristic_evaluation(self, user_query: str, ai_response: str, context: str) -> Dict:
        """
        Fast heuristic-based evaluation when LLM fails
        Uses simple text matching and length analysis
        """
        logger.info("    Using heuristic evaluation (LLM-free)")
        
        # Basic relevance: keyword overlap
        query_words = set(user_query.lower().split())
        response_words = set(ai_response.lower().split())
        context_words = set(context.lower().split())
        
        query_response_overlap = len(query_words & response_words) / max(len(query_words), 1)
        response_context_overlap = len(response_words & context_words) / max(len(response_words), 1)
        
        # Relevance: based on keyword overlap
        relevance = min(int(query_response_overlap * 15), 10)
        
        # Completeness: based on response length relative to query
        length_ratio = len(ai_response) / max(len(user_query), 1)
        completeness = min(int(length_ratio * 3), 10)
        
        # Hallucination: inverse of context overlap (higher overlap = less hallucination)
        hallucination = min(int(response_context_overlap * 12), 10)
        
        # Factual accuracy: similar to hallucination score
        factual_accuracy = hallucination
        
        # Ensure minimum scores
        relevance = max(relevance, 4)
        completeness = max(completeness, 4)
        hallucination = max(hallucination, 5)
        factual_accuracy = max(factual_accuracy, 5)
        
        return {
            'relevance_score': relevance,
            'completeness_score': completeness,
            'hallucination_score': hallucination,
            'factual_accuracy_score': factual_accuracy,
            'issues': ['Heuristic evaluation used (LLM timeout)'],
            'missing_info': [],
            'hallucinations': [],
            'latency_ms': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost_usd': 0.0,
            'evaluation_mode': 'heuristic'
        }
    
    def evaluate_conversation(self, vector_file: str, chat_file: str) -> Dict:
        """Evaluate entire conversation"""
        logger.info(f"Loading files...")
        
        vector_data = JSONParser.load_json(vector_file)
        chat_data = JSONParser.load_json(chat_file)
        
        if not vector_data or not chat_data:
            raise Exception("Failed to load input files")
        
        conversations = ConversationExtractor.extract_conversations(chat_data)
        # CRITICAL: Use minimal context to prevent timeouts
        context = ContextExtractor.extract_context(vector_data, max_contexts=2, max_chars=600)
        
        if not conversations:
            raise Exception("No conversations found to evaluate")
        
        if not context:
            logger.warning("No context found - using empty context")
            context = "No context available"
        
        logger.info(f"Starting evaluation of {len(conversations)} turns with {len(context)} char context")
        
        results = []
        total_latency = 0
        total_input_tokens = 0
        total_output_tokens = 0
        failed_count = 0
        heuristic_count = 0
        
        for idx, conv in enumerate(conversations, 1):
            logger.info(f"Evaluating turn {conv['turn']} ({idx}/{len(conversations)})...")
            
            eval_result = self.evaluate_response(
                conv['user_query'],
                conv['ai_response'],
                context
            )
            
            # Track evaluation modes
            if eval_result.get('evaluation_mode') == 'heuristic':
                heuristic_count += 1
            if 'error' in eval_result or eval_result.get('evaluation_mode') == 'heuristic':
                failed_count += 1
                if 'error' not in eval_result:
                    logger.info(f"    Used heuristic evaluation")
            else:
                logger.info(f"    ✓ Completed ({eval_result.get('evaluation_mode', 'standard')} mode)")
            
            results.append({
                'turn': conv['turn'],
                'user_query': conv['user_query'][:200] + '...' if len(conv['user_query']) > 200 else conv['user_query'],
                'ai_response': conv['ai_response'][:200] + '...' if len(conv['ai_response']) > 200 else conv['ai_response'],
                'evaluation': eval_result
            })
            
            total_latency += eval_result.get('latency_ms', 0)
            total_input_tokens += eval_result.get('input_tokens', 0)
            total_output_tokens += eval_result.get('output_tokens', 0)
        
        avg_scores = self._calculate_averages(results)
        
        return {
            'chat_id': chat_data.get('chat_id', 'unknown'),
            'user_id': chat_data.get('user_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model,
            'total_turns_evaluated': len(results),
            'successful_evaluations': len(results) - failed_count,
            'failed_evaluations': failed_count,
            'heuristic_evaluations': heuristic_count,
            'llm_evaluations': len(results) - heuristic_count,
            'average_scores': avg_scores,
            'total_latency_ms': round(total_latency, 2),
            'avg_latency_per_turn_ms': round(total_latency / len(results), 2) if results else 0,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'estimated_cost_usd': 0.0,
            'detailed_results': results
        }
    
    def _calculate_averages(self, results: List[Dict]) -> Dict:
        """Calculate average scores from successful evaluations only"""
        if not results:
            return {}
        
        scores = {
            'relevance': [],
            'completeness': [],
            'hallucination': [],
            'factual_accuracy': []
        }
        
        for r in results:
            eval_data = r['evaluation']
            # Only include if not an error
            if 'error' not in eval_data:
                scores['relevance'].append(eval_data.get('relevance_score', 0))
                scores['completeness'].append(eval_data.get('completeness_score', 0))
                scores['hallucination'].append(eval_data.get('hallucination_score', 0))
                scores['factual_accuracy'].append(eval_data.get('factual_accuracy_score', 0))
        
        if not scores['relevance']:
            return {
                'relevance': 0,
                'completeness': 0,
                'hallucination': 0,
                'factual_accuracy': 0,
                'overall': 0
            }
        
        averages = {k: round(sum(v) / len(v), 2) for k, v in scores.items()}
        averages['overall'] = round(sum(averages.values()) / len(averages), 2)
        
        return averages


def main():
    """Main execution function"""
    import sys
    
    vector_file = sys.argv[1] if len(sys.argv) > 1 else "json_files/sample_context_vectors-01.json"
    chat_file = sys.argv[2] if len(sys.argv) > 2 else "json_files/sample-chat-conversation-01.json"
    
    logger.info("="*70)
    logger.info("LLM EVALUATION PIPELINE (Optimized)")
    logger.info("="*70)
    
    try:
        evaluator = LLMEvaluator()
        results = evaluator.evaluate_conversation(vector_file, chat_file)
        
        output_file = f"evaluation_results_{results['chat_id']}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Chat ID: {results['chat_id']}")
        print(f"User ID: {results['user_id']}")
        print(f"Model: {results['model_used']}")
        print(f"Turns Evaluated: {results['total_turns_evaluated']}")
        print(f"  • LLM Evaluations: {results['llm_evaluations']}")
        print(f"  • Heuristic Fallbacks: {results['heuristic_evaluations']}")
        print(f"  • Failed: {results['failed_evaluations']}")
        print(f"\nAverage Scores (0-10):")
        for metric, score in results['average_scores'].items():
            bar = '█' * int(score) + '░' * (10 - int(score))
            print(f"  {metric.replace('_', ' ').title():20s}: {score:5.2f} [{bar}]")
        print(f"\nPerformance:")
        print(f"  Total Latency: {results['total_latency_ms']:.2f}ms")
        print(f"  Avg Latency/Turn: {results['avg_latency_per_turn_ms']:.2f}ms")
        print(f"  Total Tokens: {results['total_tokens']}")
        
        if results['heuristic_evaluations'] > 0:
            print(f"\n⚠ Note: {results['heuristic_evaluations']} evaluation(s) used heuristic mode due to timeouts.")
            print(f"  Consider using a faster model (phi:2.7b) or reducing context size.")
        
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())