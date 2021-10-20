import numpy as np
from numba import jit

class SOM():
    """
    My SOM Implementation. The focus of this implementation is in vectorisation and \
    setting up the classes to be compatible with Numba through the static method.
    
    Usage example:
        np.random.seed(seed=42)
        input_data = np.random.random((20,3))
        size = 10
        som = SOM(size,size)
        som.train(input_data, number_iterations=10, initial_learningrate=0.1)
    """
    def __init__(self, width, height):
        super(SOM, self).__init__()
        self.width = width
        self.height = height
        
    @staticmethod
    @jit(nopython=True)
    def calculate_euclidean_distance(matrix, vector):
        """
        L2 Norm Calculation by subtracting a single row from a column matrix. 
        """
        dist_array = np.sum((matrix - vector)**2, axis=1)
        return dist_array
        
    @staticmethod
    @jit(nopython=True)
    def calculate_BMU(current_input_vector, weights):
        """
        Best Matching Unit (BMU) calculation.
        - Reshapes into column array
        - Calculates L2 Norm
        - Finds argmin id and 2D coordinates of the BMU
        """
        weights_reshaped = weights.reshape(-1,3)
        dist_array = np.sum((weights_reshaped - current_input_vector)**2, axis=1)
        min_id = np.argmin(dist_array)
        
        BMU_x_coords = min_id // weights.shape[0]
        BMU_y_coords = min_id % weights.shape[1]
        BMU_coord = np.array([BMU_x_coords,BMU_y_coords])
        return min_id, BMU_coord   
    
    @staticmethod
    @jit(nopython=True)
    def calculate_decay(current_iteration, time_constant):
        """
        Decay calculation to run once at every time step.
        """
        decay = np.exp(-current_iteration / time_constant)
        return decay
    
    @staticmethod
    @jit(nopython=True)
    def calculate_radius(initial_radius,decay):
        """
        Radius calculation to run once at every time step.
        """
        current_radius = initial_radius * decay
        return current_radius

    @staticmethod
    @jit(nopython=True)
    def calculate_neighbourhood(current_radius, BMU_coord, dimensions):
        """
        Neighbour calculation:
        - Create 3D array where the first page is the x coordinate associated with \
        each element and the second page is the y coordinate. This allows us to get \
        the x and y coordinate of any particular grid element (i.e. (3,10))
        - Subtract BMU from this git to get manhattan distance of every element to BMU coords
        - Calculate L2 norm of entire grid from BMU coords
        - Check inside radius
        - Returns coordinates and IDs of nodes inside the current_radius
        """
        # Normally we would just use np.indicies but numba doesn't like it! so we are reworking from source code
        # Create coordinates grid
        N = 2 # 2D Iamge
        output_shape = np.array((N,dimensions[0],dimensions[1])) ###
        my_tuple = (output_shape[0],output_shape[1],output_shape[2])
        coords_list = np.ones(my_tuple,dtype=np.int32)
        for i, dim in enumerate(dimensions):
            if i == 0:
                reshape_shape = (dim,1)
            else:
                reshape_shape = (1,dim)
            idx_temp = np.arange(dim)
            idx = idx_temp.reshape(reshape_shape)
            coords_list[i] = idx
        coords_list = coords_list.reshape(2,-1).T
        
        # Subtract the BMU_coord coords to get manhattans
        dist = coords_list - np.array((BMU_coord[0],BMU_coord[1]))
        abs_dist = np.abs(dist) # Absolute value of manhattan
        dist_sum = np.sqrt(np.square(abs_dist).sum(1)) # L2 Norm
        # dist_sum = (abs_dist).sum(1) # L1 Norm
        id_mask = dist_sum <= int(current_radius) # Get true/false mask
        ids = np.where(id_mask)[0] 
        return coords_list, id_mask, ids # Returns coordinates of nodes inside the current_radius

    @staticmethod
    @jit(nopython=True)
    def calculate_learningrate(initial_learningrate, decay):
        """
        Learning rate calculation to run once at every time step.
        """
        current_learningrate = initial_learningrate * decay
        return current_learningrate

    @staticmethod
    @jit(nopython=True)
    def calculate_influence(coords_list, id_mask, ids, BMU_coord, current_radius):
        """
        Vectorised influence array calculation:
        - Calculate influence using the coords list of nodes inside the radius (i.e. (22,3))
        - Pad the influence array to the full size and fill with zeros (i.e. (100,3)) to assist in update calc
        """
        dist_array = np.sum((coords_list[id_mask] - BMU_coord)**2, axis=1)
        influence_array = np.exp(- dist_array / (2 * current_radius**2))
        padded_influence_array = np.zeros((coords_list.shape[0],1))
        influence_array = np.reshape(influence_array, (-1, 1))
        padded_influence_array[ids] = influence_array
        return padded_influence_array

    @staticmethod
    @jit(nopython=True)
    def calculate_update(weights, current_learningrate, padded_influence_array, current_input_vector):
        """
        Vectorised update calculation:
        - Use the padded influence array (i.e. 100,3) to calculate updates for all weights in a \
        single vectorised equation
        """
        weights_reshaped = weights.reshape(-1,3)
        updated_weights = weights_reshaped + current_learningrate * padded_influence_array * (current_input_vector - weights_reshaped)
        updated_weights = updated_weights.reshape(weights.shape)
        return updated_weights

    @staticmethod
    @jit(nopython=True)
    def get_current_input_vector(input_data):
        """
        Randomly select a single row from input data
        """
        number_of_rows = input_data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=1)
        current_input_vector = input_data[random_indices, :]
        return current_input_vector

    @staticmethod
    @jit(nopython=True)
    def get_image(weights, size, number_attributes):
        """
        Get image for displaying the weights image
        """
        current_weights_image = np.reshape(weights, (size,size,number_attributes))
        return current_weights_image
    
    def train(self, input_data, number_iterations, initial_learningrate):
        """
        Training Loop:
        Initialise Data and Weights
        ## for 0 to n_iters
            ## Randomly choose current input data
            ## Calculate BMU
            ## Calculate Radius
            ## Calculate Neighbourhood
            ## Calculate Learning Rate
            ## Calculate Influence
            ## Calculate Update Weights
        ## Generate Image
        """


        self.input_data = input_data
        self.number_attributes = input_data.shape[1]
        self.weights = np.random.random((self.width, self.height, self.number_attributes))
        self.number_iterations = number_iterations
        self.initial_radius = np.max((self.width,self.height))/2
        self.initial_learningrate = initial_learningrate
        self.time_constant = self.number_iterations/np.log(self.initial_radius)
#         self.current_weights_image = np.random.random((self.width, self.height, self.number_attributes))
        
        for current_iteration in range(self.number_iterations):
            current_input_vector = SOM.get_current_input_vector(self.input_data)
            min_id, BMU_coord = SOM.calculate_BMU(current_input_vector, self.weights)
            decay = SOM.calculate_decay(current_iteration, self.time_constant)
            current_radius = SOM.calculate_radius(self.initial_radius, decay)
            dimensions = np.array([self.width, self.height])
            coords_list, id_mask, ids = SOM.calculate_neighbourhood(current_radius, BMU_coord, dimensions)
            current_learningrate = SOM.calculate_learningrate(self.initial_learningrate, decay)
            padded_influence_array = SOM.calculate_influence(coords_list, id_mask, ids, BMU_coord, current_radius)
            updated_weights = SOM.calculate_update(self.weights, current_learningrate, padded_influence_array, current_input_vector)
            self.weights = updated_weights
#             current_weights_image = SOM.get_image(self.weights, self.width, self.number_attributes)