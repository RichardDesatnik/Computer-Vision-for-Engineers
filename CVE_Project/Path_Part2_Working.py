from os import close
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np

class Node(object):
    def __init__(self, pose):
        self.pose = np.array(pose)
        self.x = pose[0]
        self.y = pose[1]
        self.g_value = 0
        self.h_value = 0
        self.f_value = 0
        self.parent = None

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        return (self.pose == other.pose).all()

class AStar(object):
    def __init__(self, map_path):
        self.map_path = map_path
        self.map = self.load_map(self.map_path).astype(int)
        #print(self.map)
        self.resolution = 0.05
        self.y_dim = self.map.shape[0]
        self.x_dim =self.map.shape[1]
        print(f'map size ({self.x_dim}, {self.y_dim})')

    def load_map(self, path):
        #return np.load(path)
        return np.genfromtxt(path, delimiter = ",")

    def reset_map(self):
        self.map = self.load_map(self.map_path)

    def heuristic(self, current, goal):
        return np.linalg.norm(current.pose - goal.pose)

    def get_successor(self, node):
        successor_list = []
        x,y = node.pose
        pose_list = [[x+1, y+1], [x, y+1], [x-1, y+1], [x-1, y],
                        [x-1, y-1], [x, y-1], [x+1, y-1], [x+1, y]]

        for pose_ in pose_list:
            x_, y_ = pose_
            if 0 <= x_ < self.y_dim and 0 <= y_ < self.x_dim and self.map[x_, y_] == 0:
                self.map[x_, y_] = -1
                successor_list.append(Node(pose_))
        
        return successor_list
    
    def calculate_path(self, node):
        path_ind = []
        path_ind.append(node.pose.tolist())
        current = node
        while current.parent:
            current = current.parent
            path_ind.append(current.pose.tolist())
        path_ind.reverse()
        print(f'path length {len(path_ind)}')
        path = list(path_ind)

        return path

    def plan(self, start_ind, goal_ind):
        # initialize start node and goal node class
        start_node = Node(start_ind)
        goal_node = Node(goal_ind)

        # Calculate initial h and f values for start_node
        start_node.h_value = self.heuristic(start_node, goal_node)
        start_node.f_value = start_node.g_value + start_node.h_value

        # Reset map and initialize open and closed lists
        self.reset_map()
        open_list = []
        closed_list = []
        heappush(open_list, start_node)

        # Reset map
        self.reset_map()

        # Initially, only the start node is known.
        # This is usually implemented as a min-heap or priority queue rather than a hash-set.
        # Please refer to https://docs.python.org/3/library/heapq.html for more details about heap data structure
        open_list = []
        closed_list = np.array([])
        heappush(open_list, start_node)

        # while open_list is not empty
        while len(open_list):
            # Current is the node in open_list that has the lowest f value
            # This operation can occur in O(1) time if open_list is a min-heap or a priority queue
            
            # Get the node with the lowest f value from open_list
            current = heappop(open_list)
            closed_list = np.append(closed_list, current)

            self.map[current.x, current.y] = -1

            # if current is goal_node: calculate the path by passing through the current node
            # exit the loop by returning the path
            if current == goal_node:
                print('reach goal')
                return self.calculate_path(current)
            
            for successor in self.get_successor(current):
                # Set current node as the parent of the successor
                successor.parent = current

                # Calculate g, h, and f values for the successor
                distance = np.linalg.norm(successor.pose - current.pose)
                successor.g_value = current.g_value + distance
                successor.h_value = self.heuristic(successor, goal_node)
                successor.f_value = successor.g_value + successor.h_value

                heappush(open_list, successor)

        # If the loop is exited without return any path
        # Path is not found
        print('path not found')
        return None

    def run(self, cost_map, start_ind, goal_ind):

        if cost_map[start_ind[0], start_ind[1]] == 0 and cost_map[goal_ind[0], goal_ind[1]] == 0:
            return self.plan(start_ind, goal_ind)

        else:
            print('already occupied')


def visualize_path(cost_map, path, title):
    x = [item[0] for item in path]
    x = x[1:-1]
    y = [item[1] for item in path]
    y = y[1:-1]

    plt.imshow(np.transpose(cost_map))
    plt.plot(path[0][0], path[0][1], 'x', color = 'r', label = 'start', markersize = 10)
    plt.plot(path[-1][0], path[-1][1], 'o', color = 'r', label = 'goal', markersize = 10)
    plt.scatter(x, y, label = 'path', s = 1)
    plt.legend()
    plt.title(title)
    plt.show() 

if __name__ == "__main__":
    costmap3 = np.genfromtxt('generated_binary_map.csv', delimiter = ',')

    start_ind3 = [48, 176]
    goal_ind3 = [450, 350]

    Planner3 = AStar('generated_binary_map.csv')

    path_ind3 = Planner3.run(costmap3, start_ind3, goal_ind3)
    visualize_path(costmap3, path_ind3, 'Path Planning for Robots')

    # Load the original image
    #image_path = "predict_custom.jpg"
    image_path = "predict_final.jpg"
    image = cv2.imread(image_path)

    # Example path coordinates (replace with your actual coordinates)
    # Coordinates should be in the format [(x1, y1), (x2, y2), ..., (xn, yn)]
    path_coordinates = path_ind3
    # Draw the path on the image
    for i in range(len(path_coordinates) - 1):
        start_point = path_coordinates[i]
        end_point = path_coordinates[i + 1]
        color = (255, 0, 0)  # Blue color for the path
        thickness = 2        # Line thickness
        cv2.line(image, start_point, end_point, color, thickness)

    # Draw a cross at the starting point
    start_point = path_coordinates[0]
    cross_color = (0, 0, 255)  # Red color for the cross
    cross_size = 10  # Size of the cross
    cv2.line(image, (start_point[0] - cross_size, start_point[1] - cross_size),
             (start_point[0] + cross_size, start_point[1] + cross_size), cross_color, thickness)
    cv2.line(image, (start_point[0] - cross_size, start_point[1] + cross_size),
             (start_point[0] + cross_size, start_point[1] - cross_size), cross_color, thickness)

    # Draw a solid circle at the ending point
    end_point = path_coordinates[-1]
    circle_color = (0, 255, 0)  # Green color for the circle
    circle_radius = 8  # Radius of the circle
    cv2.circle(image, end_point, circle_radius, circle_color, -1)  # -1 fills the circle

    legend_height = 100
    legend_width = 250
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

    # Add legend elements
    cv2.putText(legend, "Legend:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(legend, (10, 40), (50, 40), (255, 0, 0), 2)  # Path line (blue)
    cv2.putText(legend, "Path", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.line(legend, (10, 60), (20, 70), (0, 0, 255), 2)  # Cross (red)
    cv2.line(legend, (20, 60), (10, 70), (0, 0, 255), 2)
    cv2.putText(legend, "Start", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(legend, (15, 90), 8, (0, 255, 0), -1)  # Circle (green)
    cv2.putText(legend, "End", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Overlay the legend onto the original image
    legend_x = 950
    legend_y = 300
    #image[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = legend

    # Save and display the result
    output_path = "path_drawn_on_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Path drawn and saved to {output_path}")

