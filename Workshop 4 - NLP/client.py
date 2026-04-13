#!/usr/bin/env python
"""Cliente para el servidor de clasificación de emociones.

Uso:
    python client.py "I'm so happy today!"
    python client.py "I'm so happy today!" --url https://<id>-8081.cloudspaces.litng.ai
    python client.py --batch tweets.txt
"""
import argparse
import requests


def predict_one(url: str, text: str) -> None:
    resp = requests.post(f"{url}/predict", json={"text": text})
    resp.raise_for_status()
    result = resp.json()
    print(f"
Tweet: {result['text']}
")
    for p in result["predictions"]:
        bar = "█" * int(p["score"] * 40)
        print(f"  {p['label']:<12} {p['score']:.2%}  {bar}")


def predict_batch(url: str, filepath: str) -> None:
    with open(filepath) as f:
        texts = [line.strip() for line in f if line.strip()]

    resp = requests.post(f"{url}/predict/batch", json={"texts": texts})
    resp.raise_for_status()

    print(f"{"Tweet":<50}  {"Emoción":<12}  {"Score":>6}")
    print("-" * 74)
    for item in resp.json()["results"]:
        top     = item["predictions"][0]
        preview = (item["text"][:47] + "...") if len(item["text"]) > 50 else item["text"].ljust(50)
        print(f"{preview}  {top['label']:<12}  {top['score']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion classifier client")
    parser.add_argument("text",    nargs="?", help="Texto a clasificar")
    parser.add_argument("--batch", metavar="FILE", help="Archivo con un tweet por línea")
    parser.add_argument("--url",   default="http://localhost:8081", help="URL del servidor")
    args = parser.parse_args()

    if args.batch:
        predict_batch(args.url, args.batch)
    elif args.text:
        predict_one(args.url, args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()