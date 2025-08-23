from llama_cpp import Llama

MODEL_PATH = r"C:\Users\pc\OneDrive\Masaüstü\chatbottarimproject\Chatbottarim\llama-2-7b-chat.Q8_0.gguf"
llm = Llama(model_path=MODEL_PATH, n_threads=8)

def get_answer(question, temperature=0.25):
    
    prompt = (
    "Sen yalnızca Türkçe konuşan, tarım konusunda uzman bir chatbot'sun.\n"
    "Soruları cevaplarken sadece **doğru ve kanıtlanmış bilgileri** ver.\n"
    "Sadece Türkçe cevap ver. Kısa, net ve doğru cevaplar ver.\n"
    "Gerçek saat veya tarih bilgisini yazma, sadece tarım sorularına cevap ver.\n"
    "İngilizce veya başka dil kullanma.\n\n"
    "Soru: Domatesi ne zaman sulamalıyım?\n"
    "Cevap: Domatesleri sabah erken saatlerde sulamalısınız, toprağın nemine dikkat edin.\n\n"
    "Soru: Hangi gübreyi kullanmalıyım?\n"
    "Cevap: Domates için azot ve potasyum dengeli gübre uygundur.\n\n"
    f"Soru: {question}\nCevap:"
)
    
    try:
        result = llm(
            prompt=prompt,
            max_tokens=100,
            temperature=temperature,
            stop=["\n"]  
        )
        return result['choices'][0]['text'].strip()
    except Exception as e:
        return "Üzgünüm, şu anda cevap veremiyorum."

# Örnek
if __name__ == "__main__":
    print(get_answer("Domates yetiştirirken hangi gübreyi kullanmalıyım?"))
