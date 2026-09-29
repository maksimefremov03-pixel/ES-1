import clips
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from matplotlib.patches import Rectangle
import itertools


def load_knowledge_base(filename="knowledge_base.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            kb = json.load(f)

        for key in list(kb.keys()):
            if key.startswith("cpt_"):
                if isinstance(kb[key], dict):
                    new_dict = {}
                    for k, v in kb[key].items():
                        if isinstance(k, str):
                            new_key = tuple(k.split(','))
                        elif isinstance(k, list):
                            new_key = tuple(k)
                        else:
                            new_key = k
                        new_dict[new_key] = v
                    kb[key] = new_dict

        if "tree_nodes" not in kb:
            kb["tree_nodes"] = {"input": [], "middle": [], "output": []}
        if "tree_edges" not in kb:
            kb["tree_edges"] = []
        if "questions" not in kb:
            kb["questions"] = []
        if "node_labels" not in kb:
            kb["node_labels"] = {}
        if "costs" not in kb:
            kb["costs"] = {}
        if "cost_names" not in kb:
            kb["cost_names"] = {}

        return kb
    return None


class ExpertSystemCLIPS:
    def __init__(self, kb):
        self.env = clips.Environment()
        self.kb = kb
        self.setup()

    def setup(self):
        self.env.build("(deftemplate component (slot name (type SYMBOL)) (slot state (type SYMBOL)))")
        self.env.build("(deftemplate probability (slot event (type SYMBOL)) (slot value (type FLOAT)))")
        self.env.build("(deftemplate system (slot status (type SYMBOL)))")
        self.env.build(
            "(defrule system-init (not (system (status ?))) => (assert (system (status initializing))) (assert (system (status ready))))")

        self.load_cpts()
        self.load_performance_rule()
        self.load_results_rule()
        self.load_costs_rule()

    def load_cpts(self):
        for key, cpt in self.kb.items():
            if key.startswith("cpt_") and key != "cpt_performance":
                event_name = key.replace("cpt_", "")
                parents = self.get_parents(event_name)
                if len(parents) >= 2 and len(next(iter(cpt.keys()))) == len(parents):
                    for states, prob in cpt.items():
                        conditions = " ".join(
                            [f"(component (name {parents[i]}) (state {states[i]}))" for i in range(len(parents))])
                        self.env.build(f"""(defrule {event_name}-{'-'.join(states)}
    {conditions}
    (not (probability (event {event_name})))
    =>
    (assert (probability (event {event_name}) (value {prob}))))""")

    def get_parents(self, node):
        return [e[0] for e in self.kb["tree_edges"] if e[1] == node]

    def load_performance_rule(self):
        output_nodes = self.kb["tree_nodes"]["output"]
        if not output_nodes:
            return

        output = output_nodes[0]
        parents = self.get_parents(output)

        if len(parents) < 1:
            return

        cpt_key = "cpt_performance"
        if cpt_key not in self.kb:
            return

        cpt = self.kb[cpt_key]
        combos = list(itertools.product(*[[p, "No"] for p in parents]))

        terms = []
        for i, combo in enumerate(combos):
            prob = cpt.get(combo, 0.5)
            term = f"(bind ?p{i + 1} (* " + " ".join(
                [f"(if (eq ?state{j} {p}) then ?v{j} else (- 1 ?v{j}))" for j, p in
                 enumerate(parents)]) + f" {prob}))"
            terms.append(term)

        states_bind = " ".join([f"(bind ?state{j} {p})" for j, p in enumerate(parents)])
        terms_str = "\n            ".join(terms)
        sum_terms = " ".join([f"?p{i + 1}" for i in range(len(combos))])
        prob_conditions = " ".join([f"(probability (event {p}) (value ?v{i}))" for i, p in enumerate(parents)])

        self.env.build(f"""
        (defrule calc-{output}
            {prob_conditions}
            (not (probability (event {output})))
            =>
            {states_bind}
            {terms_str}
            (bind ?perf (+ {sum_terms}))
            (assert (probability (event {output}) (value ?perf))))
        """)

    def load_results_rule(self):
        input_nodes = self.kb["tree_nodes"]["input"]
        output_nodes = self.kb["tree_nodes"]["output"]
        if not input_nodes or not output_nodes:
            return

        conditions = " ".join([f"(component (name {n}) (state ?{n.lower()}))" for n in input_nodes])
        output = output_nodes[0]
        self.env.build(
            f"(defrule show-results {conditions} (probability (event {output}) (value ?perf)) => (assert (system (status done))))")

    def load_costs_rule(self):
        input_nodes = self.kb["tree_nodes"]["input"]
        costs = self.kb.get("costs", {})

        if not input_nodes:
            return

        conditions = " ".join([f"(component (name {n}) (state ?{n.lower()}))" for n in input_nodes])
        cost_actions = " ".join(
            [f"(if (eq ?{n.lower()} {n}) then (bind ?total (+ ?total {costs.get(n, 0)})))" for n in input_nodes])

        self.env.build(
            f"(defrule show-costs {conditions} => (bind ?total 0) {cost_actions} (assert (probability (event TotalCost) (value ?total))))")

    def add_fact(self, template, **kwargs):
        fact_str = f"({template}"
        for k, v in kwargs.items():
            fact_str += f" ({k} {v})"
        fact_str += ")"
        self.env.assert_string(fact_str)

    def run(self):
        self.env.reset()
        self.env.run()

    def get_facts(self, template_name):
        facts = []
        for fact in self.env.facts():
            if fact.template.name == template_name:
                facts.append(dict(fact))
        return facts


class TreeEditorDialog:
    def __init__(self, parent, kb):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Редактор байесовской сети")
        self.dialog.geometry("800x700")
        self.kb = kb

        self.create_widgets()
        self.load_data()

        self.dialog.transient(parent)
        self.dialog.grab_set()
        parent.wait_window(self.dialog)

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="РЕДАКТИРОВАНИЕ СЕТИ", font=("Arial", 12, "bold")).pack(pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        nodes_frame = ttk.Frame(notebook)
        notebook.add(nodes_frame, text="Узлы")

        ttk.Label(nodes_frame, text="Входные узлы (через запятую):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)
        self.input_entry = ttk.Entry(nodes_frame, width=80)
        self.input_entry.pack(pady=5, padx=10)

        ttk.Label(nodes_frame, text="Промежуточные узлы (через запятую):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)
        self.middle_entry = ttk.Entry(nodes_frame, width=80)
        self.middle_entry.pack(pady=5, padx=10)

        ttk.Label(nodes_frame, text="Выходные узлы (через запятую):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)
        self.output_entry = ttk.Entry(nodes_frame, width=80)
        self.output_entry.pack(pady=5, padx=10)

        ttk.Button(nodes_frame, text="Добавить метки узлов", command=self.add_labels).pack(pady=10)

        edges_frame = ttk.Frame(notebook)
        notebook.add(edges_frame, text="Связи")

        ttk.Label(edges_frame, text="Связи (формат: узел1 -> узел2):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)

        text_frame = ttk.Frame(edges_frame)
        text_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.edges_text = tk.Text(text_frame, width=80, height=20)
        scrollbar = ttk.Scrollbar(text_frame, command=self.edges_text.yview)
        self.edges_text.configure(yscrollcommand=scrollbar.set)
        self.edges_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        questions_frame = ttk.Frame(notebook)
        notebook.add(questions_frame, text="Вопросы")

        ttk.Label(questions_frame, text="Вопросы (формат: узел | текст вопроса):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)

        q_text_frame = ttk.Frame(questions_frame)
        q_text_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.questions_text = tk.Text(q_text_frame, width=80, height=20)
        q_scrollbar = ttk.Scrollbar(q_text_frame, command=self.questions_text.yview)
        self.questions_text.configure(yscrollcommand=q_scrollbar.set)
        self.questions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        q_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        costs_frame = ttk.Frame(notebook)
        notebook.add(costs_frame, text="Стоимость")

        ttk.Label(costs_frame, text="Стоимость исправления (формат: узел | название | стоимость):", font=("Arial", 10)).pack(pady=5, anchor=tk.W, padx=10)

        c_text_frame = ttk.Frame(costs_frame)
        c_text_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.costs_text = tk.Text(c_text_frame, width=80, height=20)
        c_scrollbar = ttk.Scrollbar(c_text_frame, command=self.costs_text.yview)
        self.costs_text.configure(yscrollcommand=c_scrollbar.set)
        self.costs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        c_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(costs_frame, text="Пример: WeakCPU | Замена процессора | 8000", font=("Arial", 9), foreground="gray").pack(pady=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Сохранить", command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

    def add_labels(self):
        all_nodes = (self.input_entry.get() + "," + self.middle_entry.get() + "," + self.output_entry.get())
        nodes = [n.strip() for n in all_nodes.split(",") if n.strip()]
        for node in nodes:
            if node not in self.kb.get("node_labels", {}):
                label = simpledialog.askstring("Метка узла", f"Введите название для узла '{node}':", parent=self.dialog)
                if label:
                    self.kb["node_labels"][node] = label

    def load_data(self):
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, ", ".join(self.kb["tree_nodes"]["input"]))

        self.middle_entry.delete(0, tk.END)
        self.middle_entry.insert(0, ", ".join(self.kb["tree_nodes"]["middle"]))

        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, ", ".join(self.kb["tree_nodes"]["output"]))

        self.edges_text.delete(1.0, tk.END)
        for edge in self.kb["tree_edges"]:
            self.edges_text.insert(tk.END, f"{edge[0]} -> {edge[1]}\n")

        self.questions_text.delete(1.0, tk.END)
        for q in self.kb.get("questions", []):
            self.questions_text.insert(tk.END, f"{q['component']} | {q['text']}\n")

        self.costs_text.delete(1.0, tk.END)
        for node in self.kb["tree_nodes"]["input"]:
            name = self.kb.get("cost_names", {}).get(node, node)
            cost = self.kb.get("costs", {}).get(node, 0)
            self.costs_text.insert(tk.END, f"{node} | {name} | {cost}\n")


    def save(self):
        old_input = set(self.kb["tree_nodes"]["input"])
        old_middle = set(self.kb["tree_nodes"]["middle"])
        old_edges = set(tuple(e) for e in self.kb["tree_edges"])

        self.kb["tree_nodes"]["input"] = [x.strip() for x in self.input_entry.get().split(",") if x.strip()]
        self.kb["tree_nodes"]["middle"] = [x.strip() for x in self.middle_entry.get().split(",") if x.strip()]
        self.kb["tree_nodes"]["output"] = [x.strip() for x in self.output_entry.get().split(",") if x.strip()]

        edges = []
        for line in self.edges_text.get(1.0, tk.END).strip().split("\n"):
            if "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    from_node = parts[0].strip()
                    to_node = parts[1].strip()
                    if from_node and to_node:
                        edges.append([from_node, to_node])
        self.kb["tree_edges"] = edges

        questions = []
        for line in self.questions_text.get(1.0, tk.END).strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 2:
                    comp = parts[0].strip()
                    text = parts[1].strip()
                    if comp and text:
                        questions.append({"component": comp, "text": text, "yes": "Да", "no": "Нет"})
        self.kb["questions"] = questions

        costs = {}
        cost_names = {}
        for line in self.costs_text.get(1.0, tk.END).strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    comp = parts[0].strip()
                    name = parts[1].strip()
                    try:
                        cost = float(parts[2].strip())
                    except ValueError:
                        cost = 0
                    if comp:
                        costs[comp] = cost
                        cost_names[comp] = name
        self.kb["costs"] = costs
        self.kb["cost_names"] = cost_names

        self.rebuild_cpts(old_input, old_middle, old_edges)
        self.dialog.destroy()

    def rebuild_cpts(self, old_input, old_middle, old_edges):
        edges = self.kb["tree_edges"]
        middle = self.kb["tree_nodes"]["middle"]
        output = self.kb["tree_nodes"]["output"]
        new_input = set(self.kb["tree_nodes"]["input"])

        for node in old_input - new_input:
            for key in list(self.kb.keys()):
                if key.startswith(f"cpt_") and node in key:
                    del self.kb[key]
            self.kb["costs"].pop(node, None)
            self.kb["cost_names"].pop(node, None)
            self.kb["node_labels"].pop(node, None)
            self.kb["questions"] = [q for q in self.kb.get("questions", []) if q["component"] != node]

        for node in middle:
            parents = self.get_parents(node)
            if len(parents) >= 2:
                cpt_key = f"cpt_{node}"
                old_cpt = self.kb.get(cpt_key, {})
                new_cpt = {}
                for combo in itertools.product(*[[p, "No"] for p in parents]):
                    new_cpt[combo] = old_cpt.get(combo, 0.5)
                self.kb[cpt_key] = new_cpt

        if output:
            out_node = output[0]
            parents = self.get_parents(out_node)
            if parents:
                old_cpt = self.kb.get("cpt_performance", {})
                if old_cpt:
                    first_old_key = next(iter(old_cpt.keys()))
                    old_parents = list(first_old_key)
                    new_parents = parents

                    new_cpt = {}
                    for new_combo in itertools.product(*[[p, "No"] for p in new_parents]):
                        best_prob = 0.5
                        for old_combo, old_prob in old_cpt.items():
                            match = True
                            for i, op in enumerate(old_parents):
                                if op in new_parents:
                                    j = new_parents.index(op)
                                    if old_combo[i] != new_combo[j]:
                                        match = False
                                        break
                            if match:
                                best_prob = old_prob
                                break
                        new_cpt[new_combo] = best_prob
                    self.kb["cpt_performance"] = new_cpt
                else:
                    new_cpt = {}
                    for combo in itertools.product(*[[p, "No"] for p in parents]):
                        new_cpt[combo] = 0.5
                    self.kb["cpt_performance"] = new_cpt

        for node in new_input:
            if node not in old_input:
                self.kb["costs"].setdefault(node, 0)
                self.kb["cost_names"].setdefault(node, node)
                self.kb["node_labels"].setdefault(node, node)

    def get_parents(self, node):
        return [e[0] for e in self.kb["tree_edges"] if e[1] == node]


class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Экспертная система")
        self.root.geometry("900x750")
        self.kb = load_knowledge_base() or {"tree_nodes": {"input": [], "middle": [], "output": []}, "tree_edges": [],
                                            "questions": [], "node_labels": {}, "costs": {}, "cost_names": {}}
        self.current_question = 0
        self.answers = {}
        self.history = []
        self.create_widgets()
        self.show_start_page()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_start_page(self):
        self.clear_frame()
        ttk.Label(self.main_frame, text="ЭКСПЕРТНАЯ СИСТЕМА", font=("Arial", 16, "bold")).pack(pady=20)
        ttk.Button(self.main_frame, text="Диагностика", command=self.show_diagnostics_page).pack(pady=5)
        ttk.Button(self.main_frame, text="Дерево сети", command=self.show_tree).pack(pady=5)
        ttk.Button(self.main_frame, text="Редактировать сеть", command=self.edit_tree).pack(pady=5)
        ttk.Button(self.main_frame, text="Редактировать вероятности", command=self.show_probability_editor).pack(pady=5)
        ttk.Button(self.main_frame, text="История", command=self.show_history).pack(pady=5)
        ttk.Button(self.main_frame, text="Сохранить в JSON", command=self.save_kb).pack(pady=5)
        ttk.Button(self.main_frame, text="Загрузить из JSON", command=self.load_kb).pack(pady=5)

    def show_diagnostics_page(self):
        self.clear_frame()
        self.current_question = 0
        self.answers = {}

        if not self.kb.get("questions"):
            ttk.Label(self.main_frame, text="Нет вопросов. Добавьте вопросы в редакторе сети.",
                      font=("Arial", 12)).pack(pady=20)
            ttk.Button(self.main_frame, text="Назад", command=self.show_start_page).pack(pady=10)
            return

        ttk.Label(self.main_frame, text="ДИАГНОСТИКА", font=("Arial", 14, "bold")).pack(pady=10)
        self.question_frame = ttk.Frame(self.main_frame)
        self.question_frame.pack(pady=20)
        self.show_question()

    def show_question(self):
        for widget in self.question_frame.winfo_children():
            widget.destroy()

        questions = self.kb["questions"]
        if self.current_question < len(questions):
            q = questions[self.current_question]
            ttk.Label(self.question_frame, text=f"Вопрос {self.current_question + 1}/{len(questions)}",
                      font=("Arial", 10)).pack(pady=5)
            ttk.Label(self.question_frame, text=q["text"], font=("Arial", 12)).pack(pady=10)
            ttk.Button(self.question_frame, text=q.get("yes", "Да"), command=lambda: self.answer("1")).pack(pady=5)
            ttk.Button(self.question_frame, text=q.get("no", "Нет"), command=lambda: self.answer("2")).pack(pady=5)
        else:
            self.run_clips()

    def answer(self, choice):
        q = self.kb["questions"][self.current_question]
        comp = q["component"]
        self.answers[comp] = comp if choice == "1" else "No"
        self.current_question += 1
        self.show_question()

    def run_clips(self):
        es = ExpertSystemCLIPS(self.kb)
        es.env.reset()
        es.add_fact("system", status="initializing")
        es.add_fact("system", status="ready")
        for comp, state in self.answers.items():
            es.add_fact("component", name=comp, state=state)
        es.env.run()
        self.show_results(es)

    def explain_results(self, components, probs):
        explanations = []
        edges = self.kb["tree_edges"]
        labels = self.kb.get("node_labels", {})

        for middle_node in self.kb["tree_nodes"]["middle"]:
            prob = probs.get(middle_node, 0)
            parent_nodes = [e[0] for e in edges if e[1] == middle_node]

            if prob > 0.5:
                reasons = []
                for p in parent_nodes:
                    if components.get(p) == p:
                        reasons.append(labels.get(p, p))
                if reasons:
                    explanations.append(f"{labels.get(middle_node, middle_node)} ({prob:.1%}): {', '.join(reasons)}")
            else:
                explanations.append(f"{labels.get(middle_node, middle_node)} ({prob:.1%}): маловероятно")

        for out_node in self.kb["tree_nodes"]["output"]:
            prob = probs.get(out_node, 0)
            explanations.append(f"{labels.get(out_node, out_node)}: {prob:.1%}")

        return explanations

    def show_tree(self):
        self.clear_frame()
        ttk.Label(self.main_frame, text="ДЕРЕВО БАЙЕСОВСКОЙ СЕТИ", font=("Arial", 14, "bold")).pack(pady=10)

        G = nx.DiGraph()

        all_nodes = self.kb["tree_nodes"]["input"] + self.kb["tree_nodes"]["middle"] + self.kb["tree_nodes"]["output"]
        if not all_nodes:
            ttk.Label(self.main_frame, text="Сеть не настроена").pack(pady=20)
            ttk.Button(self.main_frame, text="Назад", command=self.show_start_page).pack(pady=10)
            return

        G.add_nodes_from(all_nodes)
        G.add_edges_from([tuple(e) for e in self.kb["tree_edges"]])

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        pos = {}
        input_nodes = self.kb["tree_nodes"]["input"]
        middle_nodes = self.kb["tree_nodes"]["middle"]
        output_nodes = self.kb["tree_nodes"]["output"]

        for i, node in enumerate(input_nodes):
            pos[node] = (i * (8 / max(len(input_nodes) - 1, 1)), 4)

        for i, node in enumerate(middle_nodes):
            pos[node] = (i * (8 / max(len(middle_nodes) - 1, 1)) + (
                    8 - (len(middle_nodes) - 1) * (8 / max(len(middle_nodes) - 1, 1))) / 2, 2.5)

        for i, node in enumerate(output_nodes):
            pos[node] = (4, 1)

        if input_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='#AED6F1', node_size=2500, node_shape='s',
                                   ax=ax)
        if middle_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=middle_nodes, node_color='#F9E79F', node_size=2500, node_shape='s',
                                   ax=ax)
        if output_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='#F1948A', node_size=3000, node_shape='s',
                                   ax=ax)

        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=25, width=2, ax=ax)

        labels = {node: self.kb.get("node_labels", {}).get(node, node) for node in all_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)

        ax.set_title("Байесовская сеть", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='#AED6F1', label='Входные'),
            Rectangle((0, 0), 1, 1, fc='#F9E79F', label='Промежуточные'),
            Rectangle((0, 0), 1, 1, fc='#F1948A', label='Выходные')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

        canvas = FigureCanvasTkAgg(fig, self.main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Button(self.main_frame, text="Назад", command=self.show_start_page).pack(pady=10)

    def edit_tree(self):
        TreeEditorDialog(self.root, self.kb)

    def show_results(self, es):
        self.clear_frame()

        canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text="РЕЗУЛЬТАТЫ", font=("Arial", 14, "bold")).pack(pady=10)

        components = {}
        for fact in es.get_facts("component"):
            components[fact["name"]] = fact["state"]

        for comp in self.kb["tree_nodes"]["input"]:
            state = components.get(comp, "No")
            label = self.kb.get("node_labels", {}).get(comp, comp)
            status = "ДА" if state == comp else "НЕТ"
            color = "red" if state == comp else "green"
            ttk.Label(scrollable_frame, text=f"{label}: {status}", font=("Arial", 11), foreground=color).pack(pady=2)

        probs = {}
        for fact in es.get_facts("probability"):
            probs[fact["event"]] = fact["value"]

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable_frame, text="ВЕРОЯТНОСТИ", font=("Arial", 12, "bold")).pack(pady=5)

        for node in self.kb["tree_nodes"]["middle"] + self.kb["tree_nodes"]["output"]:
            prob = probs.get(node, 0)
            label = self.kb.get("node_labels", {}).get(node, node)
            ttk.Label(scrollable_frame, text=f"{label}: {prob:.3f}").pack(pady=2)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable_frame, text="ГРАФИК", font=("Arial", 12, "bold")).pack(pady=5)

        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)
        all_nodes = self.kb["tree_nodes"]["middle"] + self.kb["tree_nodes"]["output"]
        labels = [self.kb.get("node_labels", {}).get(n, n) for n in all_nodes]
        values = [probs.get(n, 0) for n in all_nodes]
        colors = ['red' if v > 0.7 else 'orange' if v > 0.3 else 'green' for v in values]
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)

        chart = FigureCanvasTkAgg(fig, scrollable_frame)
        chart.get_tk_widget().pack(pady=10)

        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable_frame, text="ОБЪЯСНЕНИЕ", font=("Arial", 12, "bold")).pack(pady=5)

        explanations = self.explain_results(components, probs)
        for exp in explanations:
            ttk.Label(scrollable_frame, text=f"• {exp}", wraplength=700, font=("Arial", 10)).pack(pady=2, anchor=tk.W)

        total = probs.get("TotalCost", 0)
        ttk.Separator(scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable_frame, text="СТОИМОСТЬ", font=("Arial", 12, "bold")).pack(pady=5)

        if total > 0:
            for comp, state in components.items():
                if state == comp and comp in self.kb.get("cost_names", {}):
                    cost = self.kb["costs"].get(comp, 0)
                    if cost > 0:
                        ttk.Label(scrollable_frame,
                                  text=f"{self.kb['cost_names'][comp]}: {cost} руб.").pack(pady=2)
            ttk.Label(scrollable_frame, text=f"Общая стоимость: {total:.0f} руб.", font=("Arial", 11, "bold")).pack(
                pady=5)
        else:
            ttk.Label(scrollable_frame, text="Затраты не требуются").pack(pady=5)

        ttk.Button(scrollable_frame, text="Назад в меню", command=self.show_start_page).pack(pady=20)

        self.history.append({
            "answers": dict(self.answers),
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "total_cost": total
        })

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_probability_editor(self):
        self.clear_frame()
        ttk.Label(self.main_frame, text="РЕДАКТИРОВАНИЕ ВЕРОЯТНОСТЕЙ", font=("Arial", 14, "bold")).pack(pady=10)

        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.entries = {}

        for node in self.kb["tree_nodes"]["middle"]:
            cpt_key = f"cpt_{node}"
            if cpt_key not in self.kb:
                continue

            parents = [e[0] for e in self.kb["tree_edges"] if e[1] == node]
            if len(parents) < 2:
                continue

            frame = ttk.Frame(notebook)
            notebook.add(frame, text=self.kb.get("node_labels", {}).get(node, node))

            parent_str = ", ".join(parents)
            ttk.Label(frame, text=f"P({node} | {parent_str})", font=("Arial", 11, "bold")).pack(pady=5)

            for key_tuple, prob in self.kb[cpt_key].items():
                row = ttk.Frame(frame)
                row.pack(pady=2, fill=tk.X, padx=40)
                text = ", ".join([f"{parents[i]}={key_tuple[i]}" for i in range(len(parents))])
                ttk.Label(row, text=text, width=50, anchor=tk.W).pack(side=tk.LEFT)
                entry = ttk.Entry(row, width=10)
                entry.insert(0, str(prob))
                entry.pack(side=tk.LEFT, padx=20)
                self.entries[f"{cpt_key}|{'|'.join(key_tuple)}"] = entry

        for node in self.kb["tree_nodes"]["output"]:
            cpt_key = "cpt_performance"
            if cpt_key not in self.kb:
                continue

            parents = [e[0] for e in self.kb["tree_edges"] if e[1] == node]
            if not parents:
                continue

            frame = ttk.Frame(notebook)
            notebook.add(frame, text=self.kb.get("node_labels", {}).get(node, node))

            parent_str = ", ".join(parents)
            ttk.Label(frame, text=f"P({node} | {parent_str})", font=("Arial", 11, "bold")).pack(pady=5)

            for key_tuple, prob in self.kb[cpt_key].items():
                row = ttk.Frame(frame)
                row.pack(pady=2, fill=tk.X, padx=40)
                text = ", ".join([f"{parents[i]}={key_tuple[i]}" for i in range(len(parents))])
                ttk.Label(row, text=text, width=50, anchor=tk.W).pack(side=tk.LEFT)
                entry = ttk.Entry(row, width=10)
                entry.insert(0, str(prob))
                entry.pack(side=tk.LEFT, padx=20)
                self.entries[f"cpt_performance|{'|'.join(key_tuple)}"] = entry

        ttk.Button(self.main_frame, text="Сохранить", command=self.save_probabilities).pack(pady=10)
        ttk.Button(self.main_frame, text="Назад", command=self.show_start_page).pack(pady=5)

    def save_probabilities(self):
        try:
            for key, entry in self.entries.items():
                parts = key.split("|")
                cpt_key = parts[0]
                key_tuple = tuple(parts[1:])

                value = float(entry.get())
                if value < 0 or value > 1:
                    raise ValueError(f"Значение должно быть от 0 до 1")

                if cpt_key in self.kb:
                    self.kb[cpt_key][key_tuple] = value

            messagebox.showinfo("Успех", "Вероятности обновлены")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_history(self):
        self.clear_frame()
        ttk.Label(self.main_frame, text="ИСТОРИЯ", font=("Arial", 14, "bold")).pack(pady=10)

        if not self.history:
            ttk.Label(self.main_frame, text="История пуста").pack(pady=20)
        else:
            for i, record in enumerate(self.history):
                frame = ttk.Frame(self.main_frame, relief=tk.RIDGE, borderwidth=1)
                frame.pack(fill=tk.X, pady=5, padx=10)

                output_nodes = self.kb["tree_nodes"]["output"]
                perf = record["probabilities"].get(output_nodes[0] if output_nodes else "LowPerformance", 0)

                if perf >= 0.7:
                    status = "КРИТИЧНО"
                elif perf >= 0.3:
                    status = "УМЕРЕННО"
                else:
                    status = "НОРМА"

                ttk.Label(frame,
                          text=f"Диагностика {i + 1} | Итог: {perf:.3f} | Статус: {status} | Затраты: {record['total_cost']:.0f} руб.",
                          font=("Arial", 10)).pack(pady=5, padx=10)

        ttk.Button(self.main_frame, text="Назад", command=self.show_start_page).pack(pady=10)

    def save_kb(self):
        kb_copy = {
            "questions": self.kb.get("questions", []),
            "costs": self.kb.get("costs", {}),
            "cost_names": self.kb.get("cost_names", {}),
            "node_labels": self.kb.get("node_labels", {}),
            "tree_nodes": self.kb["tree_nodes"],
            "tree_edges": self.kb["tree_edges"]
        }

        for key, value in self.kb.items():
            if key.startswith("cpt_") and isinstance(value, dict) and value:
                kb_copy[key] = {}
                for k, v in value.items():
                    kb_copy[key][",".join(k)] = v

        with open("knowledge_base.json", 'w', encoding='utf-8') as f:
            json.dump(kb_copy, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Успех", "База знаний сохранена")

    def load_kb(self):
        kb = load_knowledge_base()
        if kb:
            self.kb = kb
            messagebox.showinfo("Успех", "База знаний загружена")
        else:
            messagebox.showerror("Ошибка", "Файл не найден")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Application()
    app.run()