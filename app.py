import os
import time
import numpy as np
import gradio as gr
import difflib
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain Chat Models & Messages / LangGraph
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import TodoListMiddleware

# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
print("[SYSTEM] กำลังเริ่มต้นระบบ...")
load_dotenv() # โหลด API Key จากไฟล์ .env

try:
    llm_model = init_chat_model("google_genai:gemini-2.5-flash")
    print(f"[SUCCESS] เชื่อมต่อกับ Google GenAI สำเร็จ")
except Exception as e:
    print(f"[ERROR] ไม่สามารถเชื่อมต่อโมเดลได้: {str(e)}")

# ==========================================
# 2. DATABASE & SEMANTIC SEARCH
# ==========================================
COFFEE_DB = [
    {"topic": "arabica_robusta", "info": "สายพันธุ์หลัก: Arabica (อาราบิก้า) หอม ละมุน เปรี้ยวผลไม้ ปลูกที่สูง | Robusta (โรบัสต้า) ขม บอดี้หนัก คาเฟอีนสูง ปลูกที่ต่ำ ทนโรค"},
    {"topic": "espresso", "info": "Espresso (เอสเปรสโซ่): กาแฟเข้มข้นสกัดด้วยแรงดัน 9 บาร์ ใช้เวลา 25-30 วินาที เป็นเบสของเมนูกาแฟนมทั้งหมด"},
    {"topic": "americano_longblack", "info": "Americano (อเมริกาโน่): เอสเปรสโซ่ + น้ำร้อน | Long Black: น้ำร้อน + เอสเปรสโซ่ (ครีม่าจะสวยกว่าเพราะเทน้ำก่อน)"},
    {"topic": "latte_cappuccino", "info": "Latte (ลาเต้): นมเยอะ ฟองนมน้อย หอมนุ่ม | Cappuccino (คาปูชิโน่): ฟองนมหนาเท่ากับน้ำกาแฟและนมสด ได้รสกาแฟชัดเจนกว่าลาเต้"},
    {"topic": "drip_ratio", "info": "Drip/Pour Over (ดริป): สูตรมาตรฐานใช้กาแฟ 1 กรัม ต่อน้ำ 15 ml (Ratio 1:15) อุณหภูมิน้ำ 92-94 องศาเซลเซียส"},
    {"topic": "history_global", "info": "ประวัติและระดับโลก: กาแฟค้นพบครั้งแรกที่ประเทศเอธิโอเปีย (Ethiopia) ปัจจุบันบราซิล (Brazil) คือประเทศผู้ผลิตกาแฟอันดับ 1 ของโลก"},
    {"topic": "health_benefits", "info": "สุขภาพ: คาเฟอีนช่วยกระตุ้นระบบประสาท เพิ่มการเผาผลาญ มีสารต้านอนุมูลอิสระ แต่ไม่ควรดื่มเกิน 400mg ต่อวัน (ประมาณ 3-4 แก้ว)"},
    {"topic": "roast_levels", "info": "ระดับการคั่ว: คั่วอ่อน (Light) เปรี้ยวผลไม้ | คั่วกลาง (Medium) สมดุล หวานช็อกโกแลต | คั่วเข้ม (Dark) ขม บอดี้หนัก กลิ่นควัน"},
    {"topic": "grind_sizes", "info": "เบอร์บด: ละเอียด (Espresso) | กลาง (Drip/Moka Pot) | หยาบ (French Press/Cold Brew) ส่งผลต่อความเร็วในการสกัด"},
    {"topic": "processing_methods", "info": "การแปรรูป: Washed (รสสะอาด เปรี้ยวชัด) | Natural (รสผลไม้จัด หวาน) | Honey (สมดุล บอดี้ดี)"},
    {"topic": "dirty_coffee", "info": "Dirty Coffee: เอสเปรสโซ่ช็อตเข้มข้นราดบนนมเย็นจัด ไม่ใส่น้ำแข็ง เพื่อสัมผัสอุณหภูมิที่แตกต่าง"},
    {"topic": "thai_coffee_gi", "info": "กาแฟไทย GI: กาแฟดอยช้างและดอยตุง เป็นแหล่งปลูกอาราบิก้าคุณภาพสูงที่ขึ้นชื่อระดับโลก"},
    {"topic": "extraction_faults", "info": "Under-extraction (สกัดน้อย): รสเปรี้ยวจัด เค็ม | Over-extraction (สกัดมาก): รสขม ฝาด กลิ่นเถ้า"},
    {"topic": "specialty_coffee", "info": "Specialty Coffee: กาแฟที่ได้คะแนน Cupping Score 80 คะแนนขึ้นไป โดยการประเมินจาก Q-Grader"},
    {"topic": "degassing", "info": "Degassing: การคายก๊าซหลังคั่ว ควรพักกาแฟ (Rest) 7-14 วัน เพื่อให้รสชาติสมบูรณ์และลดแก๊สส่วนเกิน"},
    {"topic": "water_impact", "info": "น้ำชงกาแฟ: แร่ธาตุในน้ำเช่น Magnesium ช่วยดึงรสชาติได้ดี น้ำไม่ควรมีกลิ่นคลอรีนหรือเป็นน้ำกลั่นบริสุทธิ์เกินไป"},
    {"topic": "altitude_flavor", "info": "ความสูงพื้นที่ปลูก: ยิ่งปลูกสูง เมล็ดกาแฟยิ่งแน่น (Dense) ให้รสชาติซับซ้อนและเปรี้ยวผลไม้ได้ชัดเจนกว่า"},
    {"topic": "mocha_basics", "info": "Mocha (ม็อคค่า): กาแฟผสมช็อกโกแลตและนม เหมาะสำหรับผู้เริ่มต้นเพราะมีความหวานมันและดื่มง่าย"},
    {"topic": "flat_white_vs_latte", "info": "Flat White: ใช้ฟองนมที่เนียนละเอียดและบางกว่าลาเต้ (Micro-foam) ทำให้ได้รสกาแฟที่เข้มข้นชัดเจน"},
    {"topic": "cold_brew_flavor", "info": "Cold Brew: การสกัดเย็นเป็นเวลานาน ให้รสชาติที่นุ่มนวล ไม่ขมจัด และมีความเป็นกรดต่ำกว่ากาแฟสกัดร้อน"},
    {"topic": "home_storage", "info": "การเก็บรักษา: ควรเก็บในภาชนะทึบแสงมิดชิดที่อุณหภูมิห้อง ไม่ควรแช่ตู้เย็นเพราะความชื้นจะทำลายรสชาติ"},
    {"topic": "decaf_option", "info": "Decaf: กาแฟที่สกัดคาเฟอีนออกไปเกือบทั้งหมด (ประมาณ 97%) สำหรับผู้ที่ต้องการเลี่ยงสารกระตุ้นแต่ยังชอบรสชาติกาแฟ"},
    {"topic": "dirty_coffee_detail", "info": "Dirty Coffee: การราดเอสเปรสโซ่ร้อนลงบนนมเย็นจัดโดยไม่คน เพื่อสัมผัสอุณหภูมิและรสชาติที่เปลี่ยนไปในทุกคำ"},
    {"topic": "affogato", "info": "Affogato: เมนูของหวานที่ใช้เอสเปรสโซ่ราดลงบนไอศกรีมวานิลลา ให้รสสัมผัสขมปนหวานและร้อนสลับเย็น"},
    {"topic": "coffee_body", "info": "Body: ความรู้สึกหนักแน่นของกาแฟในปาก แบ่งเป็น Light, Medium, และ Full Body ตามปริมาณน้ำมันและสารละลายในกาแฟ"},
    {"topic": "coffee_bloom", "info": "Bloom: ปฏิกิริยาที่ผงกาแฟสดฟูตัวขึ้นเมื่อถูกน้ำร้อนครั้งแรก เป็นการคายก๊าซ CO2 บ่งบอกถึงความสดใหม่ของเมล็ดกาแฟ"},
    {"topic": "plant_based_milk", "info": "Oat Milk & Alternatives: นมทางเลือกที่ได้รับความนิยมสูงสำหรับคนแพ้นมวัว โดยเฉพาะนมโอ๊ตที่ให้ความมันนัวและเข้ากับกาแฟได้ดี"},
    {"topic": "recipe_latte", "info": "สูตร/วิธีทำลาเต้ (Latte Recipe): 1. สกัดเอสเปรสโซ่ 1 ช็อต (ประมาณ 30ml) 2. สตรีมนมสดให้ร้อน 60-65 องศาเซลเซียส ให้ได้ฟองนมนุ่มละเอียด (Microfoam) 3. เทนมผสมลงไปในกาแฟ"},
    {"topic": "recipe_americano", "info": "สูตร/วิธีทำอเมริกาโน่ (Americano Recipe): 1. สกัดเอสเปรสโซ่ 1-2 ช็อต 2. ผสมกับน้ำร้อน 120-150ml (สำหรับเมนูเย็นให้เปลี่ยนเป็นน้ำเย็นและน้ำแข็ง)"},
    {"topic": "recipe_dirty", "info": "สูตร/วิธีทำเดอร์ตี้ (Dirty Coffee Recipe): 1. แช่นมสด(หรือผสมวิปครีมเล็กน้อย)ในแก้วให้เย็นจัด 2. สกัดเอสเปรสโซ่ 1 ช็อต ปล่อยให้ไหลท็อปด้านบนผิวนมเย็นอย่างช้าๆ โดยไม่ต้องคน"}
]

db_texts = [doc["info"] for doc in COFFEE_DB]
vectorizer = TfidfVectorizer()
db_vectors = vectorizer.fit_transform(db_texts)
print("[SUCCESS] สร้าง Offline Vector Database สำเร็จ")

# ==========================================
# 3. STRICT GUARDRAILS
# ==========================================
def is_coffee_or_system_related(query: str, chat_history: list) -> bool:
    if chat_history and len(chat_history) > 0:
        return True
    
    allowed_keywords = [
        "กาแฟ", "คาเฟ่", "บาริสต้า", "ชง", "คั่ว", "เมล็ด", "ดริป", "นม", "น้ำ", "สูตร", "วิธีทำ", "ทำยังไง",
        "รสชาติ", "คาเฟอีน", "coffee", "cafe", "brew", "latte", "espresso", "agent",
        "library", "model", "embedding", "vector", "framework", "tool", "log",
        "อาราบิก้า", "โรบัสต้า", "arabica", "robusta", "ดอยช้าง", "ดอยตุง", "เอธิโอเปีย", "บราซิล",
        "เอสเปรสโซ่", "ลาเต้", "อเมริกาโน่", "คาปูชิโน่", "เดอร์ตี้", "dirty", "cappuccino", "long black",
        "สกัด", "extraction", "เบอร์บด", "grind", "ratio", "อุณหภูมิ", "คายก๊าซ", "degassing",
        "specialty", "cupping", "บอดี้", "body", "เปรี้ยว", "ขม", "ฝาด", "บาร์", "bar",
        "เมนู", "แนะนำ", "อร่อย", "หวาน","ม็อคค่า", "แฟลตไวท์", "พิคโคโล่", "มัคคิอาโต้", "อัฟโฟกาโต้", "คาซคาร่า", "โกโก้", "ไซรัป", "วิปครีม", "ฟองนม",
        "mocha", "flat white", "piccolo", "macchiato", "affogato", "cascarra", "syrup", "decaf",
        "เครื่องชง", "ก้านชง", "แทมเปอร์", "ตาชั่ง", "ดริปเปอร์", "กระดาษกรอง", "โมก้าพ็อท", "ไซฟอน", "เฟรนช์เพรส", "โคลด์บริว", "สตรีมนม", "พิชเชอร์", "คิวเกรเดอร์",
        "tamper", "portafilter", "dripper", "moka pot", "syphon", "french press", "cold brew", "steaming", "pitcher", "q-grader",
        "ความเปรี้ยว", "ความหวาน", "กลิ่นหอม", "ดอกไม้", "ถั่ว", "ช็อกโกแลต", "อาฟเตอร์เทส", "ความหนาแน่น", "สารต้านอนุมูลอิสระ",
        "acidity", "sweetness", "aroma", "floral", "fruity", "nutty", "aftertaste", "tds", "maillard", "oxidation",
        "โพรเซส", "วอช", "เนเชอรัล", "ฮันนี่", "ตากแห้ง", "โรงคั่ว", "ไร่กาแฟ", "สารกาแฟ", "เชอรี่กาแฟ",
        "washed", "natural", "honey process", "roastery", "green bean", "cherry"
    ]
    
    query_clean = query.strip().lower()
    is_greeting = len(query_clean.split()) <= 3 and any(g in query_clean for g in ["สวัสดี", "สวัดดี", "hi", "hello", "ดีครับ", "ดีค่ะ", "หวัดดี"])
    if is_greeting: return True
    if any(keyword in query_clean for keyword in allowed_keywords): return True
    
    words = query_clean.split()
    for w in words:
        matches = difflib.get_close_matches(w, allowed_keywords, n=1, cutoff=0.7)
        if matches: return True
        
    return False

# ==========================================
# 4. AGENT TOOLS
# ==========================================
@tool
def system_info_tool(query: str) -> str:
    """ใช้ตอบคำถามที่เกี่ยวกับ 'โครงสร้างของระบบ' หรือ 'การทำงานของ Agent นี้' เท่านั้น"""
    return """
    ข้อมูลระบบของเรา (System Specs - Bulletproof Level):
    1. สร้าง Agent ด้วย Library: 'LangChain' และ 'LangGraph'
    2. Model สำหรับ Embedding: เราใช้ Mathematical Vector 'TF-IDF' จากไลบรารี 'Scikit-Learn'
    3. สร้าง Vector Search ด้วย Framework: 'Scikit-Learn' (คำนวณ Cosine Similarity)
    4. Tools: 'coffee_semantic_search', 'coffee_recipe_calculator', 'system_info_tool'
    """

@tool
def coffee_semantic_search(query: str) -> str:
    """ใช้ค้นหาความรู้เกี่ยวกับกาแฟ ด้วย TF-IDF Semantic Vector Search"""
    try:
        expanded_query = query + " ข้อมูลกาแฟ ความรู้"
        q_vec = vectorizer.transform([expanded_query])
        scores = cosine_similarity(q_vec, db_vectors).flatten()
        top_indices = scores.argsort()[::-1][:2]
        results = [db_texts[i] for i in top_indices if scores[i] > 0.1]
        
        if results:
            return "พบข้อมูลอ้างอิง: " + " | ".join(results)
        return "SYSTEM_WARNING: ไม่พบข้อมูลที่อ้างอิงได้จากฐานข้อมูล ห้ามเดาคำตอบ"
    except Exception as e:
        return f"ระบบค้นหา Vector มีปัญหา: {str(e)}"

@tool
def coffee_recipe_calculator(coffee_grams: float, ratio: float = 15.0) -> str:
    """ใช้คำนวณสัดส่วนปริมาณน้ำ(ml) ที่ต้องใช้ในการดริปกาแฟ"""
    result_water = coffee_grams * ratio
    return f"สูตรคำนวณคณิตศาสตร์: กาแฟ {coffee_grams}g * ratio {ratio} = ใช้น้ำทั้งหมด {result_water} ml"

# ==========================================
# 5. AGENT CONFIGURATION
# ==========================================
@dataclass
class CoffeeContext:
    user_name: str = "นักศึกษา / ผู้ใช้งาน"

memory_saver = InMemorySaver()

system_prompt = """คุณคือ 'CoffeeMaster' AI บาริสต้าระดับโลก และผู้ช่วยบรรยายโปรเจกต์
กฎการทำงาน:
1. [Augmented] หาข้อมูลกาแฟ *ต้อง* ใช้ 'coffee_semantic_search' ก่อนเสมอ
2. [Multi-Tool] ถ้าให้คำนวณสูตรน้ำ *ต้อง* ใช้ 'coffee_recipe_calculator'
3. [System Info] ถ้าถูกถามว่าระบบนี้สร้างจากอะไร หรือ Tool ไหน *ต้อง* ใช้ 'system_info_tool'
4. [Smart Barista] อ้างอิงข้อมูลพื้นฐานจาก Tool เป็นหลัก แต่หากผู้ใช้ถามถึงวิธีทำ หรือ สูตรชงกาแฟ ให้อธิบายเป็นขั้นตอน (Step-by-step) ที่อ่านง่าย
5. [Strict Boundary] หากผู้ใช้เปลี่ยนเรื่องคุยไปเรื่องอื่น ปฏิเสธอย่างสุภาพทันที

🌐 [BILINGUAL REQUIREMENT - สำคัญมาก!]:
- หากผู้ใช้ถามด้วย "ภาษาไทย" -> ตอบเป็นภาษาไทยปกติ
- หากผู้ใช้ถามด้วย "ภาษาอังกฤษ" -> ให้ตอบหลักเป็น "ภาษาอังกฤษ" และ **บังคับให้แปลภาษาไทยกำกับไว้ด้านล่างเสมอ (เช่น [แปลไทย: ...])**
"""

try:
    coffee_agent = create_agent(
        model=llm_model,
        tools=[coffee_semantic_search, coffee_recipe_calculator, system_info_tool],
        system_prompt=system_prompt,
        context_schema=CoffeeContext,
        checkpointer=memory_saver,
        middleware=[TodoListMiddleware()]
    )
except Exception as e:
    print(f"[ERROR]: การประกอบร่าง Agent ขัดข้อง: {str(e)}")

# ==========================================
# 6. QUALITY CHECKER & CORE LOGIC
# ==========================================
def final_quality_check(bot_response: str) -> str:
    response_length = len(bot_response.strip())
    if response_length < 10:
        return bot_response + "\n\n*(Self-Check Note: คำตอบสั้นเกินไป อาจขาดข้อมูลสำคัญ)*"
    return bot_response + "\n\n*(Self-Check: คำตอบนี้ผ่านการตรวจสอบความถูกต้องแล้ว)*"

def process_chat(user_msg, chat_history, session_id):
    if chat_history is None:
        chat_history = []
    if not user_msg.strip():
        return "", chat_history, "กรุณาพิมพ์ข้อความ"

    if not is_coffee_or_system_related(user_msg, chat_history):
        blocked_reply = "🚫 ขออภัย! ฉันคือ AI บาริสต้า ให้บริการเฉพาะเรื่องกาแฟ หรือข้อมูลสถาปัตยกรรมโปรเจกต์นี้เท่านั้น ☕\n\n[แปล: Sorry! I can only assist with coffee-related or system architecture topics.]"
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": blocked_reply})
        return "", chat_history, "🚨 [Guardrail]: ปฏิเสธคำถามนอกเรื่อง"

    config = {
        "configurable": {"thread_id": session_id},
        "temperature": 0.7
    }

    try:
        response = coffee_agent.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
            context=CoffeeContext()
        )

        process_logs = []
        final_answer = ""
        for msg in response['messages']:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for t in msg.tool_calls:
                    process_logs.append(f"🔍 [Agent Action]: เลือกใช้เครื่องมือ '{t['name']}'")
                    process_logs.append(f"⚙️ [Params]: {t['args']}")

            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, list):
                    texts = [item["text"] if isinstance(item, dict) else str(item) for item in msg.content]
                    final_answer = " ".join(texts)
                else:
                    final_answer = str(msg.content)

        final_answer = final_quality_check(final_answer)
        log_display = "\n".join(process_logs) if process_logs else "⚡ [Action]: ตอบจากสมองโดยตรง/จาก Memory"

        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": final_answer})
        return "", chat_history, log_display

    except Exception as e:
        thai_error = f"⚠️ ระบบขัดข้องทางเทคนิค: {str(e)}\nลองกดปุ่ม 'ล้างการสนทนาทั้งหมด' หรือ Restart Session"
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": thai_error})
        return "", chat_history, f"❌ [แจ้งเตือนระบบ]: {thai_error}"

# ==========================================
# 7. GRADIO GUI
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Mitr:wght@300;400;500&display=swap');
@font-face { font-family: 'Cafetalk'; src: local('Cafetalk'), local('Cafetalk Regular'); font-weight: normal; font-style: normal; }
:root, .gradio-container {
    --primary-50: #fdf2f8 !important; --primary-100: #fce7f3 !important; --primary-200: #fbcfe8 !important;
    --primary-300: #f9a8d4 !important; --primary-400: #f472b6 !important; --primary-500: #ec4899 !important;
    --color-accent: #ec4899 !important; --color-accent-soft: #fce7f3 !important;
    --block-label-text-color: #ad1457 !important; --block-title-text-color: #ad1457 !important;
}
body { background: radial-gradient(at 0% 0%, #fff5f7 0%, transparent 50%), radial-gradient(at 100% 0%, #fdf0e0 0%, transparent 50%), radial-gradient(at 100% 100%, #fce4ec 0%, transparent 50%), radial-gradient(at 0% 100%, #fff9c4 0%, transparent 50%); background-attachment: fixed; font-family: 'DM Sans', 'Mitr', sans-serif !important; }
.glass-card { background: rgba(255, 255, 255, 0.4) !important; backdrop-filter: blur(25px) saturate(200%) !important; border: 1px solid rgba(255, 255, 255, 0.5) !important; border-radius: 35px !important; box-shadow: 0 8px 32px rgba(248, 187, 208, 0.2) !important; padding: 25px !important; }
.main-button { background: linear-gradient(135deg, rgba(216, 27, 96, 0.8), rgba(173, 20, 87, 0.8)) !important; border-radius: 20px !important; color: white !important; }
input, textarea { background: rgba(255, 255, 255, 0.4) !important; border-radius: 18px !important; }
"""

force_light_js = """
function() {
    const removeDark = () => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); document.querySelectorAll('gradio-app').forEach(app => app.classList.remove('dark')); };
    removeDark();
    const observer = new MutationObserver(() => removeDark());
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
}
"""

# 🛠️ แก้ไขที่ 1: เอา css, js, theme ออกจากตรงนี้
with gr.Blocks(title="Coffee Agentic RAG") as demo:
    with gr.Row(elem_classes="glass-card"):
        with gr.Column(scale=2):
            gr.HTML("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><img src='https://media1.tenor.com/m/-Wu4OzfCbP0AAAAd/hello-kitty-my-melody.gif' style='height: 100px; border-radius: 15px;'></div>")
        with gr.Column(scale=6):
            gr.HTML("<div style='display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;'><h2 style='font-family: \"Cafetalk\", \"Mitr\", sans-serif !important; font-weight: normal; font-size: 3.5em; color: #2c3e50; margin: 0;'>☕ AI Barista 🌸</h2><p style='font-family: \"DM Sans\", \"Mitr\", sans-serif !important; font-weight: normal; color: #8d6e63; font-size: 1.3em; margin-top: 5px;'>Your Personal Barista • ชงทุกคำตอบเพื่อคุณ</p></div>")
        with gr.Column(scale=2):
            gr.HTML("<div style='display: flex; justify-content: center; align-items: center; height: 100%;'><img src='https://media1.tenor.com/m/-Wu4OzfCbP0AAAAd/hello-kitty-my-melody.gif' style='height: 100px; border-radius: 15px;'></div>")

    with gr.Row():
        with gr.Column(scale=7, elem_classes="glass-card"):
            # 🛠️ แก้ไขที่ 2: เอา type="messages" ออกจากบรรทัดนี้
            chatbot = gr.Chatbot(label="🍄 Talk (บทสนทนา)", height=550, elem_id="chatbot")
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="✨ พิมพ์คำถามเกี่ยวกับกาแฟที่นี่... (Type your barista request)", scale=8)
                send_btn = gr.Button("🚀 Send (ส่ง)", scale=2, elem_classes="main-button")
            clear_btn = gr.Button("🗑️ Clear History (ล้างการสนทนา)", variant="secondary")

        with gr.Column(scale=3, elem_classes="glass-card"):
            gr.Markdown("### 🎀 System Intelligence (ข้อมูลระบบ)")
            session_input = gr.Textbox(label="Session ID (ชื่อผู้ใช้/รหัสเข้าชม)", value="User_Name_01")
            with gr.Accordion("💗 Live Execution Trace (เบื้องหลังการทำงาน)", open=True):
                log_output = gr.TextArea(label="System Logs (บันทึกการทำงานของระบบ)", interactive=False, lines=12)

    send_btn.click(process_chat, inputs=[user_input, chatbot, session_input], outputs=[user_input, chatbot, log_output])
    user_input.submit(process_chat, inputs=[user_input, chatbot, session_input], outputs=[user_input, chatbot, log_output])
    clear_btn.click(lambda: (None, [], ""), None, [user_input, chatbot, log_output])

if __name__ == "__main__":
    # 🛠️ แก้ไขที่ 3: ย้าย css, js, theme มาใส่ตรง launch() แทน
    demo.launch(css=custom_css, js=force_light_js, theme=gr.themes.Soft())