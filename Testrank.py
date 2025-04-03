import tkinter as tk
from tkinter import ttk, scrolledtext
import networkx as nx
import nltk
from nltk.corpus import stopwords
import re
import string

# Note: this code is adapted from
# https://github.com/summanlp/textrank and
# https://www.youtube.com/watch?v=KKpf0EcgkUU
# https://www.youtube.com/watch?v=2l6Fa767kEw
# https://www.nltk.org/

nltk.data.find("corpora/stopwords")
nltk.data.find("taggers/averaged_perceptron_tagger")


class TextRankApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TextRank Keyword Extraction")
        self.root.geometry("800x600")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Enter your text:").pack(anchor="w", pady=(0, 5))

        self.text_input = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.keyword_count = 8
        self.window_size = 4

        extract_button = ttk.Button(
            main_frame, text="Extract Keywords", command=self.extract_keywords
        )
        extract_button.pack(pady=(0, 10))

        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(results_frame, text="Results:").pack(anchor="w", pady=(0, 5))
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def extract_keywords(self):
        text = self.text_input.get(1.0, tk.END)
        top_n = self.keyword_count
        window_size = self.window_size

        if not text.strip():
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please enter some text to analyze.")
            return

        keyword_scores = self.textrank_keywords(text, top_n, window_size)
        self.update_text_results(text, keyword_scores)

    def update_text_results(self, text, keyword_scores):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Original text:\n")
        self.results_text.insert(tk.END, text + "\n\n")
        self.results_text.insert(tk.END, "Important words by TextRank:\n\n")
        if keyword_scores:
            scores = [score for _, score in keyword_scores]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1

            for word, score in keyword_scores:
                normalized_score = (score - min_score) / score_range
                self.results_text.insert(tk.END, f"{word:<15} {normalized_score:.4f}\n")
        else:
            self.results_text.insert(tk.END, "No keywords found.")

    def preprocess_text(self, text):
        text = text.lower()
        text = "".join(
            [
                char
                for char in text
                if char not in string.punctuation or char in ["_", "."]
            ]
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def textrank_keywords(self, text, top_n=10, window_size=3):
        processed_text = self.preprocess_text(text)
        tokens = processed_text.split()
        stop_words = set(stopwords.words("english"))
        custom_stopwords = {"to", "and", "then", "by", "them", "an", "in"}
        stop_words.update(custom_stopwords)

        filtered_tokens = [
            token for token in tokens if token not in stop_words and len(token) > 1
        ]

        if len(filtered_tokens) < 3:
            filtered_tokens = [t for t in tokens if t not in stop_words]

        graph = nx.Graph()

        for token in filtered_tokens:
            graph.add_node(token)

        for i in range(len(filtered_tokens)):
            for j in range(i + 1, min(i + window_size + 1, len(filtered_tokens))):
                if filtered_tokens[i] != filtered_tokens[j]:
                    if graph.has_edge(filtered_tokens[i], filtered_tokens[j]):
                        graph[filtered_tokens[i]][filtered_tokens[j]]["weight"] += 1.0
                    else:
                        graph.add_edge(
                            filtered_tokens[i], filtered_tokens[j], weight=1.0
                        )

        if len(filtered_tokens) > 1 and not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                graph.add_edge(node1, node2, weight=0.1)

        try:
            for node in graph.nodes():
                graph.add_edge(node, node, weight=0.1)

            personalization = {}
            for i, token in enumerate(filtered_tokens):
                personalization[token] = 1.0 - (i / (len(filtered_tokens) * 2))

            scores = nx.pagerank(graph, personalization=personalization)

            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        except Exception as e:
            print(f"PageRank calculation failed: {e}")
            word_scores = {}
            for i, token in enumerate(filtered_tokens):
                position_weight = 1.0 - (i / (len(filtered_tokens) * 2))
                if token in word_scores:
                    word_scores[token] += position_weight
                else:
                    word_scores[token] = position_weight

            total = sum(word_scores.values()) or 1.0  # no /0
            word_scores = {word: score / total for word, score in word_scores.items()}

            return sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


if __name__ == "__main__":
    root = tk.Tk()
    app = TextRankApp(root)
    root.mainloop()