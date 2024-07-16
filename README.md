<a name="readme-top"></a>



<br />
<div align="center">
  <h1 align="center">AI-based monitoring of 3D printing</h1>
  
  <p align="center">
    <a href="https://github.com/Green-AI-Hub-Mittelstand/AI-based-monitoring-of-3D-printing/issues">Report Bug</a>
    ·
    <a href="https://github.com/Green-AI-Hub-Mittelstand/AI-based-monitoring-of-3D-printing/issues">Request Feature</a>
  </p>

  <br />

  <p align="center">
    <a href="https://www.green-ai-hub.de">
    <img src="images/green-ai-hub-keyvisual.svg" alt="Logo" width="80%">
  </a>
    <br />
    <h3 align="center"><strong>Green-AI Hub Mittelstand</strong></h3>
    <a href="https://www.green-ai-hub.de"><u>Homepage</u></a> 
    | 
    <a href="https://www.green-ai-hub.de/kontakt"><u>Contact</u></a>
  
   
  </p>
</div>

<br/>

## About The Project

This repository contains the program code for the pilot project “AI-based monitoring of 3D printing”, which was developed as part of the Green-AI Hub. The program code can detect errors in 3D printing. More information about the project can be found here: [Green-AI-Hub: AI-based monitoring of 3D printing](https://www.green-ai-hub.de/pilotprojekte/pilotprojekt-swms).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Table of Contents
<details>
  <summary><img src="images/table_of_contents.jpg" alt="Logo" width="2%"></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#table-of-contents">Table of Contents</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

Clone the Git repository and install the packages from the `requirements.txt` file.

The file `inference.py` uses an existing model to predict errors. The file `download_images.py` downloads the labeled dataset. The file `train_model.py` allows training a new model with the labeled data.

For all files, paths and credentials must be adjusted in the program code.

```bash
# Clone the repository
git clone https://github.com/Green-AI-Hub-Mittelstand/AI-based-monitoring-of-3D-printing
cd AI-based-monitoring-of-3D-printing

# Install the required packages
pip install -r requirements.txt
```


## Usage

Make sure to update the paths and credentials in each script as needed before running them.


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the GNU General Public License V3. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

The software will not be activly maintained as this was developed as part of a finished pilot project. If you have any questions, please contact Daphne Theodorakopoulos (daphne.theodorakopoulos@dfki.de).

Green-AI Hub Mittelstand - info@green-ai-hub.de

Project Link: https://github.com/Green-AI-Hub-Mittelstand/repository_name

<br />
  <a href="https://www.green-ai-hub.de/kontakt"><strong>Get in touch »</strong></a>
<br />
<br />

<p align="left">
    <a href="https://www.green-ai-hub.de">
    <img src="images/green-ai-hub-mittelstand.svg" alt="Logo" width="45%">
  </a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

