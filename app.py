from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
from grid_environment import GridEnv

app = Flask(__name__)

# Load the pretrained Q-table
q_table = joblib.load('q_learning_model.pkl')
env = GridEnv(size=(5, 5))

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/get_path', methods=['POST'])
def get_path():
    data = request.json
    start = tuple(data['start'])  # Starting coordinates
    goal = tuple(data['goal'])  # Goal coordinates
    scenario = data['scenario']  # Disaster scenario
    
    # Determine the path based on the disaster scenario
    if scenario == 'fire':
        path = find_path_for_fire(start, goal)
    elif scenario == 'earthquake':
        path = find_path_for_earthquake(start, goal)
    elif scenario == 'flood':
        path = find_path_for_flood(start, goal)
    else:
        path = []

    return jsonify({'path': path})

def find_path_for_fire(start, goal):
    # Logic for finding a path in a fire scenario
    # Simulated path: straight line from start to goal
    path = [start]
    while start[0] < goal[0]:
        start = (start[0] + 1, start[1])  # Move down
        path.append(start)
    while start[1] < goal[1]:
        start = (start[0], start[1] + 1)  # Move right
        path.append(start)
    return path

def find_path_for_earthquake(start, goal):
    # Logic for finding a path in an earthquake scenario
    # Simulated path: zigzag pattern
    path = [start]
    while start[0] < goal[0]:
        start = (start[0] + 1, start[1] + (-1 if start[0] % 2 == 0 else 1))  # Zigzag movement
        path.append(start)
    return path

def find_path_for_flood(start, goal):
    # Logic for finding a path in a flood scenario
    # Simulated path: vertical then horizontal
    path = [start]
    while start[0] < goal[0]:
        start = (start[0] + 1, start[1])  # Move down
        path.append(start)
    while start[1] < goal[1]:
        start = (start[0], start[1] + 1)  # Move right
        path.append(start)
    return path

if __name__ == '__main__':
    app.run(debug=True)
