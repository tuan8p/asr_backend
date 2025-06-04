from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

device = 0 if torch.cuda.is_available() else -1

print(f"Đang load ASR model lên {'GPU' if device == 0 else 'CPU'}...")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="tuan8p/whisper-small-vi",
    device=device
)

print("ASR model đã sẵn sàng.")

print(f"Đang load NLP model lên {'GPU' if device == 0 else 'CPU'}...")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", trust_remote_code=True)
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", trust_remote_code=True, use_safetensors=True)

print("NLP model đã sẵn sàng.")

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """Chuyển câu thành vector embedding trung bình (mean pooling)."""
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_state = outputs[0]  # (1, seq_len, hidden_dim)
        sentence_embedding = last_hidden_state.mean(dim=1)  # (1, hidden_dim)
    return sentence_embedding.squeeze(0)  # (hidden_dim,)

# Mẫu câu lệnh và intent tương ứng
intent_templates = {
    "bật đèn": "TURN_ON_LIGHT",
    "không tắt đèn": "TURN_ON_LIGHT",
    "tối quá": "TURN_ON_LIGHT",
    "không thấy gì": "TURN_ON_LIGHT",
    "tối như mực": "TURN_ON_LIGHT",
    "tối rồi": "TURN_ON_LIGHT",
    "tắt đèn": "TURN_OFF_LIGHT",
    "không bật đèn": "TURN_OFF_LIGHT",
    "sáng quá": "TURN_OFF_LIGHT",
    "chói quá": "TURN_OFF_LIGHT",
    "sáng rồi": "TURN_OFF_LIGHT",
    "bật quạt": "TURN_ON_FAN",
    "tắt quạt": "TURN_OFF_FAN",
    "quạt không ngừng": "TURN_ON_FAN",
    "nóng quá": "TURN_ON_FAN",
    "hầm quá": "TURN_ON_FAN",
    "quạt ngừng": "TURN_OFF_FAN",
    "không tắt quạt": "TURN_ON_FAN",
    "không bật quạt": "TURN_OFF_FAN",
    "lạnh quá": "TURN_OFF_FAN",
    "mở cửa": "OPEN_DOOR",
    "mở khóa cửa": "OPEN_DOOR",
    "tắt khóa cửa": "OPEN_DOOR",
    "không đóng cửa": "OPEN_DOOR",
    "tôi sắp ra ngoài": "OPEN_DOOR",
    "tôi chuẩn bị về nhà": "OPEN_DOOR",
    "đóng cửa": "CLOSE_DOOR",
    "khóa cửa": "CLOSE_DOOR",
    "không mở cửa": "CLOSE_DOOR",
    "tôi ra ngoài rồi": "CLOSE_DOOR",
    "tôi vô nhà rồi": "CLOSE_DOOR",

    "bật chế độ ban đêm": "TURN_ON_LIGHT_AND_TURN_ON_FAN_AND_CLOSE_DOOR",
    "tắt chế độ ban đêm": "TURN_OFF_LIGHT_AND_TURN_OFF_FAN_AND_OPEN_DOOR",
    "bật chế độ an ninh": "CLOSE_DOOR_AND_TURN_ON_FACE_DETECTION",
    "tắt chế độ an ninh": "OPEN_DOOR_AND_TURN_OFF_FACE_DETECTION",
    "bật tất cả thiết bị": "TURN_ON_LIGHT_AND_TURN_ON_FAN_AND_OPEN_DOOR",
    "tắt tất cả thiết bị": "TURN_OFF_LIGHT_AND_TURN_OFF_FAN_AND_CLOSE_DOOR",
}

# Hàm embedding
def get_sentence_embedding(sentence: str) -> torch.Tensor:
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_state = outputs[0]
        sentence_embedding = last_hidden_state.mean(dim=1)
    return sentence_embedding.squeeze(0)

# Lưu sẵn embeddings của intent mẫu
template_embeddings = {k: get_sentence_embedding(k) for k in intent_templates.keys()}

# Trích xuất điều kiện số
def extract_numeric_condition(sentence: str) -> dict:
    patterns = [
        # Nhiệt độ
        (# vd: nhiệt độ khoảng 30 độ C
            r"(nhiệt độ|nóng|lạnh) .*? (\d+)? .*?",
            "temperature"
        ),
        (# vd: nhiệt độ 30 độ C
            r"(nhiệt độ|nóng|lạnh) (\d+)? .*?",
            "temperature"
        ),
        (# vd: nhiệt độ cao/thấp
            r"(nhiệt độ|nóng|lạnh) .*?",
            "temperature"
        ),
        (
            r"(nóng|lạnh)",
            "temperature"
        ),
        # Độ ẩm
        (# vd: độ ẩm khoảng 70%
            r"(độ ẩm|nồm|khô) .*? (\d+)?",
            "humidity"
        ),
        (# vd: độ ẩm 70%
            r"(độ ẩm|nồm|khô) (\d+)? .*?",
            "humidity"
        ),
        (# vd: độ ẩm cao/thấp/khô/ẩm/ít/nhiều
            r"(độ ẩm|nồm|khô) .*?",
            "humidity"
        ),
        (
            r"(ẩm|nồm|khô)",
            "humidity"
        ),
        # Ánh sáng
        (
            r"(sáng|tối)",
            "light"
        ),
        # Quạt
        (# vd: mức khoảng 70%
            r"(mức|tốc độ|quay) .*? (\d+)?",
            "fan"
        ),
        (# vd: mức 70%
            r"(mức|tốc độ|quay) (\d+)? .*?",
            "fan"
        ),
        (# vd: mức 1/2/3
            r"(mức|tốc độ) (\d+)?",
            "fan"
        ),
        (# vd: mức cao/thấp/vừa
            r"(mức|tốc độ|quay) .*?",
            "fan"
        ),
        (
            r"(nhanh|mạnh|cao|chậm|yếu|thấp|vừa|thường)",
            "fan"
        ),
        # Thời gian
        (
            r"(sau)\s*"
            r"(?:(?P<hour>\d+)\s*(giờ|h|g)\s*)?"
            r"(?:(?P<minute>\d+)\s*(phút|p|m)\s*)?"
            r"(?:(?P<second>\d+)\s*(giây|s))?",
            "time"
        )
    ]

    for pattern, sensor in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            val = None
            unit = ""
            op = "="

            if any(kw in sentence for kw in ["trên", "nóng", "nhiều hơn", "ẩm", "nồm", "cao"]):
                op = ">"
                if "độ ẩm" in sentence and not any(kw in sentence for kw in ["trên", "nhiều hơn", "nồm", "cao"]):
                    op = "="
            elif any(kw in sentence for kw in ["dưới", "lạnh", "ít hơn", "khô", "thấp"]):
                op = "<"
            print(f"match: {match.groups()}")
            if sensor == "time":
                unit = "seconds"
                hour = int(match.group("hour")) if match.group("hour") else 0
                minute = int(match.group("minute")) if match.group("minute") else 0
                second = int(match.group("second")) if match.group("second") else 0
                val = hour * 3600 + minute * 60 + second
            elif sensor == "temperature":
                unit = "°C"
                if len(match.groups()) > 1:
                    if match.group(2):
                        op = "="
                        val = int(match.group(2))
                        if any(kw in sentence for kw in ["độ k", "°k", "°ka", "độ ka", "độ ca", "°ca"]) and "độ khoảng" not in sentence:
                            val -= 273
                elif any(kw in sentence for kw in ["nóng", "cao"]):
                    val = 30
                elif any(kw in sentence for kw in ["lạnh", "thấp"]):
                    val = 20
            elif sensor == "humidity":
                unit = "%"
                if len(match.groups()) > 1:
                    if match.group(2):
                        op = "="
                        val = int(match.group(2))
                elif any(kw in sentence for kw in ["khô", "thấp", "ít"]):
                    val = 30
                elif "nồm" in sentence:
                    val = 90
                elif any(kw in sentence for kw in ["cao", "nhiều"]):
                    val = 70
                elif "ẩm" in sentence and "độ ẩm" not in sentence:
                    val = 70
            elif sensor == "light":
                unit = "lux"
                if "tối" in sentence:
                    op = "<"
                    val = 20
                elif "sáng" in sentence:
                    op = ">"
                    val = 30
            elif sensor == "fan":
                unit = "%"
                if len(match.groups()) > 1:
                    if match.group(2):
                        op = "="
                        val = int(match.group(2))
                        if val == 1:# 1 là mức thấp nhất
                            val = 30
                        elif val == 2:
                            val = 70
                        elif val == 3:
                            val = 100
                elif any(kw in sentence for kw in ["nhanh", "mạnh", "cao"]):
                    val = 100
                elif any(kw in sentence for kw in ["chậm", "yếu", "thấp"]):
                    val = 30
                elif any(kw in sentence for kw in ["vừa", "thường"]):
                    val = 70

            return {
                "sensor": sensor,
                "op": op,
                "value": val,
                "unit": unit
            }

    return None

# Dự đoán intent + điều kiện
def nlp_pipeline(sentence: str) -> dict:
    condition = extract_numeric_condition(sentence)
    sentence_wo_condition = re.sub(r"khi .*|nếu .*|lúc .*|khi trời .*|nếu trời .*|lúc trời .*|sau .*", "", sentence).strip()

    emb = get_sentence_embedding(sentence_wo_condition).unsqueeze(0)
    sims = {}
    for template, template_emb in template_embeddings.items():
        template_emb = template_emb.unsqueeze(0)
        sim = cosine_similarity(emb, template_emb)[0][0]
        sims[template] = sim
    best_template = max(sims, key=sims.get)

    return {
        "sentence": sentence,
        "intent": intent_templates[best_template],
        "matched_template": best_template,
        "similarity": float(sims[best_template]),
        "condition": condition
    }
