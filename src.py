from data import sohbet_et

def main():
    print("Tarım Asistanı'na hoş geldiniz! (Çıkmak için 'çık' yazın.)")
    while True:
        try:
            soru = input("Sen: ").strip()
            if soru.lower() in ("çık", "exit", "quit"):
                print("Görüşmek üzere! İyi günler.")
                break
            if not soru:
                print("Lütfen bir soru yazınız.")
                continue
            cevap = sohbet_et(soru)
            print("Bot:", cevap)
        except KeyboardInterrupt:
            print("\nProgramdan çıkılıyor. Hoşça kal!")
            break
        except Exception as e:
            print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
