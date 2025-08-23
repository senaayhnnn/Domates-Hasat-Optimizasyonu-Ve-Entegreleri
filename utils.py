import datetime

def temizle(metin: str) -> str:
    """
    Metnin başındaki ve sonundaki boşlukları temizler,
    fazla satır başı ve boşlukları azaltır.
    Örnek:
        "  Merhaba \n\n  Dünya  " -> "Merhaba\nDünya"
    """
    if not isinstance(metin, str):
        return ""
    satirlar = [satir.strip() for satir in metin.splitlines() if satir.strip()]
    return "\n".join(satirlar).strip()


def formatla_cevap(cevap: str, on_ekler=None) -> str:
    """
    Cevap içinde belirtilen ön eklerden (prefix) varsa, onlardan sonrasını alır,
    ardından temizler.
    Örnek: "Assistant: Merhaba" -> "Merhaba"
    
    Args:
        cevap (str): İşlenecek metin.
        on_ekler (list[str], optional): Temizlenecek ön ekler listesi. 
    
    Returns:
        str: Temizlenmiş metin.
    """
    if on_ekler is None:
        on_ekler = ["Bot:", "Assistant:", ""]

    for on_ek in on_ekler:
        if on_ek and on_ek in cevap:
            cevap = cevap.split(on_ek)[-1]
    return temizle(cevap)


def logla(mesaj: str):
    """
    Zamana göre terminale log basar.
    Format: [YYYY-MM-DD HH:MM:SS] mesaj
    """
    simdi = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{simdi}] {mesaj}")
