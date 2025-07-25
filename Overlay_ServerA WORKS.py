from flask import Flask, render_template_string, jsonify
import json
import os

app = Flask(__name__)

# HTML Overlay Template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <style>
    body {
      background-color: rgba(0, 0, 0, 0);
      color: white;
      font-family: Arial, sans-serif;
    }
    .overlay {
      background-color: rgba(0, 0, 0, 0.6);
      padding: 10px 15px;
      border-radius: 12px;
      max-width: 400px;
    }
    h2 {
      margin-top: 0;
      font-size: 22px;
    }
    ul {
      margin: 0;
      padding-left: 20px;
    }
    li {
      font-size: 16px;
    }
  </style>
</head>
<body>
  <div class="overlay" id="overlay">
  <h2 id="speaker-name">Loading...</h2>
  <div id="donor-section">
    <h3 style="margin-bottom: 5px; border-bottom: 1px solid #aaa; padding-bottom: 3px;">
      Top Contributors (2024)
    </h3>
    <ul id="donor-list"></ul>
  </div>

  <script>
    async function refresh() {
      try {
        const res = await fetch('/data.json', {
          cache: "no-store"
        });
        const data = await res.json();
        document.getElementById('speaker-name').textContent = data.name || "No speaker detected";
        const ul = document.getElementById('donor-list');
        ul.innerHTML = '';
        (data.top_donors || []).forEach(d => {
          const li = document.createElement('li');
          li.textContent = `${d.name} – ${d.amount}`;
          ul.appendChild(li);
        });
      } catch(e) {
        console.error("Error fetching overlay data:", e);
      }
    }

    setInterval(refresh, 3000);
    refresh();
  </script>
</body>
</html>
"""

@app.route("/")
def overlay():
    return render_template_string(HTML_TEMPLATE)

@app.route("/data.json")
def data_json():
    try:
        if os.path.exists("overlay_data.json"):
            with open("overlay_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                print("✅ Serving overlay data:", data.get("name", "No name"))
                return jsonify(data)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
    
    return jsonify({
        "name": "No speaker detected",
        "top_donors": []
    })

if __name__ == "__main__":
    print("🌐 Overlay server running at http://localhost:5014")
    app.run(host="0.0.0.0", port=5014)


@app.route("/")
def overlay():
    return render_template_string(HTML_TEMPLATE)

@app.route("/data.json")
def data_json():
    if os.path.exists("overlay_data.json"):
        with open("overlay_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"name": "No speaker", "top_donors": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5014)