# ☕ CoffeeMaster: Agentic RAG 🌸
**ระบบ AI บาริสต้าส่วนตัวอัจฉริยะ (Smart Coffee Assistant)**

โปรเจกต์นี้เป็นการพัฒนาระบบ AI Agentic RAG ที่มีความสามารถในการค้นหาข้อมูล ตอบคำถาม และคำนวณสูตรกาแฟ โดยผสานเทคโนโลยี Semantic Search และ Agent Orchestration เข้าด้วยกัน

## ✨ ฟีเจอร์หลัก (Key Features)
* **🛡️ Strict Guardrails:** ระบบคัดกรองคำถามหน้าบ้าน ป้องกันการถามนอกเรื่อง (รองรับ Typo Tolerance)
* **🧠 Contextual Memory:** AI สามารถจดจำบริบทการสนทนา เพื่อการพูดคุยที่ต่อเนื่อง
* **🛠️ Multi-Tool Agent:** AI สามารถเลือกใช้เครื่องมือที่เหมาะสมได้เอง:
  * `Semantic Search` (ค้นหาความรู้เรื่องกาแฟด้วย TF-IDF)
  * `Recipe Calculator` (คำนวณสูตรดริปกาแฟ)
  * `System Info` (ตรวจสอบสถาปัตยกรรมระบบ)
* **✅ Quality Checker:** ระบบตรวจสอบคุณภาพและความสมบูรณ์ของคำตอบก่อนแสดงผล
* **🎨 Liquid Glass GUI:** หน้าจอ Interface ที่สวยงาม ใช้งานง่าย สไตล์พาสเทล

## 🚀 สถาปัตยกรรมระบบ (System Architecture)
1. **Agent Library:** LangChain & LangGraph
2. **LLM Model:** Gemini 2.5 Flash
3. **Embedding & Vector Search:** TfidfVectorizer & Cosine Similarity (Scikit-Learn Framework)
4. **UI Framework:** Gradio

## 💡 วิธีการรันโปรแกรม (How to Run)
1. ไปที่เมนู `รันไทม์ (Runtime)` ด้านบน
2. เลือก `รันทั้งหมด (Run all)`
3. รอประมาณ 1 นาที ระบบจะแสดงหน้าต่าง UI แชทบอทขึ้นมาที่ด้านล่างสุดของโค้ด
4. หรือสามารถคลิกที่ลิงก์ `https://....gradio.live` เพื่อเปิดเต็มจอได้

---
**👥 จัดทำโดย (Team Members):**
1. ทิพวรรณ ยิ้มเนียม 1660903921
2. สิริกานต์ ปุริสังคหะ 1660904101
3. อรพินธ์ นาคุณ 1660904689
