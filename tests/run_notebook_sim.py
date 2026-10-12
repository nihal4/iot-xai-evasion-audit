"""Execute the Kaggle notebook locally with /kaggle paths remapped (smoke check)."""
import json, sys, os
nb = json.load(open(sys.argv[1])); root = sys.argv[2]; inp = sys.argv[3]
os.makedirs(root, exist_ok=True)
g = {"display": print}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"]).replace("/kaggle/working", root).replace("/kaggle/input", inp)
    if src.startswith("%%writefile"):
        first, body = src.split("\n", 1)
        path = first.split(" ", 1)[1].strip()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(body)
        continue
    src = src.replace("from IPython.display import FileLink, display", "FileLink = lambda *a, **k: a")
    exec(compile(src, "cell", "exec"), g)
print("NOTEBOOK OK")
