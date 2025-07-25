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
    h3 {
      margin: 10px 0 4px;
      font-size: 18px;
      border-bottom: 1px solid white;
      padding-bottom: 3px;
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

    <div id="donors-section">
      <h3>Top Contributors</h3>
      <ul id="donor-list"></ul>
    </div>

    <div id="industries-section" style="margin-top:10px;">
      <h3>Top Industries</h3>
      <ul id="industry-list"></ul>
    </div>

    <div id="networth-section" style="margin-top:10px;">
      <h3>Est. Net Worth (2018)</h3>
      <p id="networth">Loading...</p>
    </div>
  </div>
  <script>
    async function refresh() {
      try {
        const res = await fetch('/data.json', { cache: "no-store" });
        const data = await res.json();

        document.getElementById('speaker-name').innerHTML = data.name.replace(/\\n/g, "<br>");

        const donorUl = document.getElementById('donor-list');
        donorUl.innerHTML = '';
        (data.top_donors || []).forEach(d => {
          const li = document.createElement('li');
          li.textContent = `${d.name} - ${d.amount.replace(/^\$/, "")}`;
          donorUl.appendChild(li);
        });

        const industryUl = document.getElementById('industry-list');
        industryUl.innerHTML = '';
        (data.top_industries || []).forEach(i => {
          const li = document.createElement('li');
          li.textContent = `${i.name} - ${i.amount.replace(/^\$/, "")}`;
          industryUl.appendChild(li);
        });

        // Net worth
        const nw = data.networth_2018 || "N/A";
        document.getElementById('networth').textContent = nw;

      } catch(e) {
        console.error(e);
      }
    }

    setInterval(refresh, 5000);
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
                app.logger.info("Serving overlay data: %s", data.get("name", "No name"))
                return jsonify(data)
    except Exception as e:
        app.logger.error("❌ Error reading JSON: %s", e)

    return jsonify({
        "name": "No speaker detected",
        "networth_2018": "N/A",
        "top_donors": [],
        "top_industries": []
    })

if __name__ == "__main__":
    print("Overlay server running at http://localhost:5014")
    app.run(host="0.0.0.0", port=5014)