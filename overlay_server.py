from flask import Flask, render_template_string
import json
import os

app = Flask(__name__)

# HTML Overlay Template with transparent background
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
  <div class="overlay">
    <h2>{{ name }}</h2>
    <ul>
    {% for donor in top_donors %}
      <li>{{ donor.name }} – {{ donor.amount }}</li>
    {% endfor %}
    </ul>
  </div>
</body>
</html>
"""
@app.route("/")
def overlay():
    if os.path.exists("overlay_data.json"):
        with open("overlay_data.json", "r", encoding="utf-8") as f:
            overlay_data = json.load(f)
    else:
        overlay_data = {"name": "No speaker detected", "top_donors": []}
    return render_template_string(HTML_TEMPLATE, **overlay_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5014)
