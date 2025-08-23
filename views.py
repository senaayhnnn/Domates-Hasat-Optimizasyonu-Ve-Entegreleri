from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
import logging
from datetime import datetime
from llama_cpp import Llama  # Llama model kütüphanesi

# Logger ayarla
logger = logging.getLogger(__name__)

# Model dosyasının tam yolu
MODEL_PATH = r"C:\Users\pc\OneDrive\Masaüstü\chatbottarimproject\Chatbottarim\llama-2-7b-chat.Q8_0.gguf"

# Llama modeli yükle (bir kez, global)
llm = Llama(model_path=MODEL_PATH, n_threads=8)

@csrf_exempt
def chat(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Sadece POST kabul edilir'}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Geçersiz JSON'}, status=400)

    user_message = data.get('message', '').strip()

    if not user_message:
        return JsonResponse({'error': 'Mesaj boş olamaz'}, status=400)

    logger.info(f"User mesajı: {user_message}")

    # Chatbot cevabı üret
    reply = chatbot_generate_reply(user_message)

    logger.info(f"Bot cevabı: {reply}")

    response = {
        'reply': reply,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    return JsonResponse(response)


def chatbot_generate_reply(message: str) -> str:
    """
    Llama modeli ile tarım chatbotu için cevap üretir.
    Chatbotun tarım alanında olduğunu ve tamamen Türkçe doğru bilgilerle cevap vermesi gerektiğini belirtir.
    """

    # Promptu string olarak tanımla ve örnek sorular ekle
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
        f"Soru: {message}\n"
        "Cevap:"
    )

    try:
        response = llm(
            prompt=prompt,
            max_tokens=100,  # Daha uzun cevaplar için artırılabilir
            temperature=0.25,
            stop=["\n", "User:", "Kullanıcı:", "Assistant:", "Asistan:"]
        )
        reply = response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Chatbot cevap üretme hatası: {e}")
        reply = "Üzgünüm, şu anda cevap veremiyorum. Lütfen daha sonra tekrar deneyin."

    return reply


def agroculus_view(request):
    """
    Agroculus için basit bir sayfa render edelim.
    Burada istersen template'e veri de gönderebilirsin.
    """
    context = {
        'title': 'Agroculus Ana Sayfa',
        'description': 'Tarım chatbotu ve diğer modüller için ana sayfa.'
    }
    return render(request, 'agroculus.html', context)
