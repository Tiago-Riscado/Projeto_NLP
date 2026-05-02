PROMPTS = {
    "simples": """Classifica a emoção principal do texto. 
Responde apenas com:
sadness, joy, love, anger ou fear.

Texto: "{text}"
""",
    "avancado": """Analisa cuidadosamente a emoção expressa no texto.
Considera sarcasmo, emojis e linguagem informal.

Emoções possíveis:
sadness, joy, love, anger, fear

Responde apenas com uma palavra.

Texto: "{text}"
""",
    "conservador": """Classifica a emoção dominante do texto.
Em caso de dúvida, escolhe a emoção principal.

Responde apenas com:
sadness, joy, love, anger ou fear.

Texto: "{text}"
"""
}
