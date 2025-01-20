# object-tracking
This repository is designed to integrate various object tracking algorithms and related functionality. 
It aims to support multiple projects involving object detection and tracking, 
with ongoing updates to include advanced tracking methods.

## Supported Tracking Algorithms
- **SORT (Simple Online and Realtime Tracking)**: Efficient and widely used tracking algorithm implemented in this repository.
- **DeepSORT (Deep Learning-based SORT)**: Advanced version of SORT that leverages appearance features for robust tracking.

## Installation

To set up the environment and run the project:
1. **Operating System**: Ubuntu 22.04
2. **Python Version**: Python 3.10.x
3. **Virtual Environment**: [Conda](https://docs.conda.io/en/latest/)
    ```bash
    conda create -n object-tracking python=3.10.x
    conda activate object-tracking
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## References
- **SORT Algorithm**: Original implementation by [abewley](https://github.com/abewley/sort).
- **DeepSORT Algorithm**: Original implementation by [nwojke](https://github.com/nwojke/deep_sort).
  - **Pytorch Version**: [ModelBunker](https://github.com/ModelBunker/Deep-SORT-PyTorch)