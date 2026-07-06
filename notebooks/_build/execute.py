import sys, time, nbformat
from nbclient import NotebookClient
name = sys.argv[1]
path = f"notebooks/{name}"
nb = nbformat.read(path, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=180, kernel_name="python3",
                        resources={"metadata": {"path": "notebooks"}})
client.execute()
nbformat.write(nb, path)
n_png = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"{name}: {time.time()-t0:.1f}s, figures={n_png}, errors={n_err}")
