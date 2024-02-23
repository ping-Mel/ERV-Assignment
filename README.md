
# ERA V2 Assignment 5

## Description

This project, Assignment 5 for ERA V2, focuses on leveraging PyTorch for defining a Convolutional Neural Network (CNN) model. It introduces utils.py to encapsulate functions, providing a streamlined and efficient approach for using Jupyter Notebook. The objective is to import the CNN model architecture and encapsulate functions within utils.py to efficiently display the model's training results


## Installation

To get started with this project, follow these steps:

```bash
git clone https://github.com/ping-Mel/ERV-Assignment.git
cd ERV-Assignment
pip install -r requirements.txt
```

Ensure you have Python 3.x installed along with all the necessary dependencies listed in `requirements.txt`.

## Usage

Here's a quick example to get you started:

```python
# Import the model class from your project
from model import Net
import utils

# Instantiate and use your model
device = utils.assign_compute()
model = Net().to(device)
```

For more detailed usage, refer to the `s5.ipynb` notebook which contains comprehensive examples and tutorials on how to utilize the models and utilities provided in this project to train the first neual network.

## Files Description

- `model.py`: Contains the main machine learning model architecture and training logic.
- `util.py`: Provides utility functions for data processing, model evaluation, and other helper functions.
- `s5.ipynb`: A Jupyter notebook that demonstrates the application, training, and evaluation of the AI model with examples.

## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the ERA V2 License.

## Contact

For any queries or further discussions, feel free to contact me at ping.xianping.wu@gmail.com or open an issue in the repository.
