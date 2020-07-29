import pandas as pd
import math
import sys
import os

# Calculate the entropy value of a split
def entropy(groups, classes):
	# Count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	entropy_val = 0
	for group in groups:
		size = len(group)

		# Avoid division by zero
		if size == 0:
			continue

		# Score the group based on the score of each class
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			if p > 0 :
				score = (p * math.log(p,2))

		# Weight the group score by its entrpy gain
		entropy_val -= score * (size/n_instances)
	return entropy_val

def try_split(index, value, data):
	left, right = list(), list()
	for row in data:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


def find_split_point(data):
	class_values = list(set(row[-1] for row in data))
	b_index, b_value, best_score, b_groups = 999, 999, 1, None
	for index in range(len(data[0])-1):
		for row in data:
			groups = try_split(index, row[index], data)
			entropy_val = entropy(groups, class_values)
			if entropy_val < best_score:
				b_index, b_value, best_score, b_groups = index, row[index], entropy_val, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_split_size, depth):
	left, right = node['groups']
	del(node['groups'])

	# Check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	
	# Stop by max depth, for better convergence
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	
	# Left child
	if len(left) <= min_split_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = find_split_point(left)
		split(node['left'], max_depth, min_split_size, depth+1)
	
	# Right child
	if len(right) <= min_split_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = find_split_point(right)
		split(node['right'], max_depth, min_split_size, depth+1)

def build_tree(train, max_depth, min_split_size):
	root = find_split_point(train)
	split(root, max_depth, min_split_size, 1)
	return root

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('Provided arguments not enough')
	else:
		train_data_path = sys.argv[1]
		test_data_path = sys.argv[2]

		# Read in data
		train_df = pd.read_csv(train_data_path, header=None)
		test_df = pd.read_csv(test_data_path, header=None)

		# Convert data frames to arrays
		train_data = train_df.values
		test_data = test_df.values

		MAX_DEPTH = 8
		MIN_SPLIT_SIZE = 6

		dt = build_tree(train_data, MAX_DEPTH, MIN_SPLIT_SIZE)

		output_name = os.path.basename(train_data_path).split("_")[0] + '_predictions.csv'
		writer = open(output_name, 'w')

		for row in test_data:
			prediction = predict(dt, row)
			writer.write('%d\n' % prediction)
		writer.close()

