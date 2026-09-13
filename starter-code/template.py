"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from openai import OpenAI
from tools import TOOL_DEFINITIONS, TOOL_MAP

SYSTEM_PROMPT = """Bạn là trợ lý AI hỗ trợ khách hàng Vingroup. Bạn có các công cụ sau:
{tools}

TUÂN THỦ NGHIÊM NGẶT quy trình sau để xử lý yêu cầu:
Thought: Suy nghĩ xem cần làm gì tiếp theo dựa trên thông tin hiện có.
Action: {{"name": "<tên_tool>", "args": {{<các_tham_số>}}}}
Observation: <Hệ thống sẽ trả về kết quả ở đây>

(Lặp lại quy trình Thought -> Action -> Observation cho đến khi đủ dữ liệu)

Khi đã đủ dữ liệu, hoặc nếu câu hỏi là FAQ không cần tra cứu:
Thought: Tôi đã có đủ thông tin để trả lời.
Final Answer: <Câu trả lời hoàn chỉnh, chi tiết cho khách hàng>

LƯU Ý QUAN TRỌNG:
- TUYỆT ĐỐI KHÔNG tự bịa đặt dữ liệu (chuyến bay, thời tiết...). Phải dùng Action để tra cứu.
- Chỉ output Action dưới dạng chuẩn JSON.
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"

    def query(self, user_input: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_input}],
                temperature=0.7,
                max_tokens=256,
            )
            return {
                "status": "success",
                "tool_calls": [],
                "answer": response.choices[0].message.content
            }
        except Exception as exc:
            return {
                "status": "error",
                "tool_calls": [],
                "answer": f"[OpenAI unavailable: {exc}]"
            }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.trace = []

    def _build_system_prompt(self) -> str:
        tools_desc = json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        return SYSTEM_PROMPT.format(tools=tools_desc)

    def _parse_action(self, text: str) -> dict | None:
        """Thuật toán đếm ngoặc nhọn để trích xuất JSON an toàn (vượt qua lỗi regex greedy)"""
        action_idx = text.find('Action:')
        if action_idx == -1: return None
        json_start = text.find('{', action_idx)
        if json_start == -1: return None
        
        brace_count = 0
        for i in range(json_start, len(text)):
            if text[i] == '{': brace_count += 1
            elif text[i] == '}': brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[json_start:i+1])
                except json.JSONDecodeError:
                    return None
        return None

    def _extract_section(self, text: str, section_name: str) -> str:
        """Hàm helper để trích xuất Thought và Final Answer"""
        pattern = rf'{section_name}:\s*(.+?)(?=(?:Action:|Observation:|Final Answer:|$))'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        tool_name = tool_name.strip().lower()
        if tool_name not in TOOL_MAP:
            return f"Error: Tool '{tool_name}' không tìm thấy."
        
        try:
            result = TOOL_MAP[tool_name](**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def run(self, user_input: str) -> dict:
        self.trace = []
        conversation = [{"role": "user", "content": user_input}]
        system_msg = {"role": "system", "content": self._build_system_prompt()}
        
        for iteration in range(1, self.max_iterations + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_msg] + conversation,
                    temperature=0.5, 
                    max_tokens=512,
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                return {"status": "error", "iterations": iteration, "trace": self.trace, "answer": f"API Error: {str(e)}"}

            thought = self._extract_section(response_text, "Thought")
            final_answer = self._extract_section(response_text, "Final Answer")
            action = self._parse_action(response_text)

            current_trace = {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": None,
                "final_answer": final_answer if final_answer else None
            }
            self.trace.append(current_trace)

            if final_answer:
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "trace": self.trace,
                    "answer": final_answer
                }

            if action and "name" in action and "args" in action:
                observation = self._execute_tool(action["name"], action["args"])
                self.trace[-1]["observation"] = observation
                
                conversation.append({"role": "assistant", "content": response_text})
                conversation.append({"role": "user", "content": f"Observation: {observation}"})
                
                if sum(1 for t in self.trace if t.get("observation") and not t["observation"].startswith("Error")) >= 2:
                    conversation.append({
                        "role": "user", 
                        "content": "Nếu bạn đã có đủ thông tin từ Observation, hãy xuất Final Answer."
                    })
            else:
                conversation.append({"role": "assistant", "content": response_text})
                conversation.append({
                    "role": "user",
                    "content": "Observation: Missing or invalid Action format. Hãy sử dụng Action (chuẩn JSON) nếu cần tra cứu, hoặc Final Answer nếu đã có đủ thông tin."
                })

        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "trace": self.trace,
            "answer": "Hệ thống không thể hoàn thành yêu cầu trong số bước cho phép."
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    result_base = chatbot.query(user_query)
    print("Status:", result_base["status"])
    print("Answer:", result_base["answer"])
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result_react = agent.run(user_query)
    print("Status:", result_react["status"])
    print("Iterations:", result_react["iterations"])
    print("Answer:", result_react["answer"])
    print("Trace Log:", json.dumps(result_react["trace"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()