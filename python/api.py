from flask import Flask, request, jsonify
import sys
sys.path.append('../build/Release')  #  .pyd file
import hnsw_module

app = Flask(__name__)
index = hnsw_module.HNSW(16, 5, 200)

@app.route('/insert', methods=['POST'])
def insert():
    data = request.json
    id = data['id']
    vector = data['vector']
    index.insert(id, vector)
    return jsonify({"status": "inserted", "id": id})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    vector = data['vector']
    k = data.get('k', 10)
    results = index.search(vector, k)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)