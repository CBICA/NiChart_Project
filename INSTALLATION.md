# Installing NiChart

We provide both a locally installable **desktop application** and a **cloud-based application**.

The [NiChart cloud application](https://neuroimagingchart.com/portal), hosted via Amazon Web Services (AWS), deploys scalable infrastructure which hosts the *NiChart* tools as a standard web application accessible via the user’s web browser. **No installation is needed**, but it requires you to upload your data to the cloud-based NiChart server for us to process it. We do not access or use your data for any other purpose than to run your requested processing and/or provide support to you, and we regularly delete user data after inactivity. However, we recognize that data privacy agreements and related concerns may nevertheless restrict use of the cloud application. If that applies to you, we suggest that you install the desktop application. Below we provide detailed installation instructions.

In particular, if you don't have a GPU on your device, the cloud application is probably the easiest way for you to use the NiChart tools.

The cloud and desktop applications are unified at the code level through the use of the Python library [Streamlit](https://streamlit.io). Consequently, the user experience is nearly identical between the cloud and desktop applications.

**Desktop installation**: Installing the desktop application currently requires [Docker](https://www.docker.com/get-started/) to be installed, as this greatly simplifies deployment and distribution of our algorithms without requiring extensive dependency management. Follow the instructions to install Docker (or Docker Desktop, on Windows/Mac) for your platform, then restart your device before continuing. We recommend having at least 20 GB of free space on your device before installing NiChart.


## Windows Instructions

Windows users will likely need to first [install the Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). 

On Windows, Docker is distributed as "Docker Desktop", an application which manages Docker on your system. 

### Docker-based Installation

#### Getting started

First, open Docker Desktop. You can do this from the start/search menu or by clicking the Desktop shortcut if you selected that during installation.

You should go into the settings using the gear icon on the top right, go to "General", and enable the settings "Use the WSL 2 based engine" and "Expose daemon on tcp://localhost:2375 without TLS" if they aren't already enabled. Then go to the "Resources" tab and disable Resource Saver if it is enabled. You should also see a green indicator on the bottom left which says "Engine running". If it's yellow or says something else, you need to wait for the service to start. Otherwise, you may need to troubleshoot your Docker installation. 

#### Choose a path to store results

In this installation, NiChart runs inside a container, which is isolated from the rest of your computer to improve security. To have data persist across sessions, you need to designate a location on your computer to store this data.

First, identify a path you want to use. In this demo we'll use "C:/Users/NiChart_Developer/Desktop/DEMODATA", but yours will vary as you can choose any folder you like. On Windows, you can navigate to a folder, then click "copy path" in the file explorer to get your path.

**Please make sure that the path you use does not already contain important data**. NiChart will not try to delete existing data, but it is good practice to select a new, empty folder.

Write down your path (for example, copy & paste it into Notepad).

#### Running the installer

Make sure you are connected to the internet in order to download the application. Then, open a terminal.

(On Windows, search "terminal", open the application that looks like a black box with a white ">_" in it. At the top of the window that appears will be a tab indicating Windows Powershell. Stay on this tab for the rest of the instructions.)

Then run this command, **making sure to replace** DATA_DIR with the data path you chose earlier:
```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install_nichart_docker_windows.ps1 DATA_DIR --distro Ubuntu
```
(Note that if you chose a different distribution for your WSL installation, you can designate that with --distro in the command above. Just replace "Ubuntu" with whatever you chose.)

This command might take a while to finish while it downloads the NiChart tools.

#### Running the application

To start NiChart, double-click the NiChart shortcut which the installer created on your desktop. It should launch your browser automatically. If not, open your browser and go to http://localhost:8501 . If you see the NiChart survey, NiChart is successfully installed.

#### Updating

To update NiChart, just run the installer again the same way you ran it above and the newest NiChart components will be installed. 

To save space, you may want to clean up your Docker images to remove older tool versions. For more information on managing Docker images, see the [Docker image docs](https://docs.docker.com/reference/cli/docker/image/).

## Linux Instructions

You will need to install Docker first and restart to make sure services are running properly.

First identify a data path where you want to persist NiChart data. We'll call that ${DATA_DIR}.

You will want to download and run the linux installer script. To do this, cd to your desired install directory and run the below. (Remember to set ${DATA_DIR} or replace it with your desired path.)

```
wget https://raw.githubusercontent.com/CBICA/NiChart_Project/main/installers/install_nichart_docker_linux.sh
chmod +x install_nichart_docker_linux.sh
./install_nichart_docker_linux.sh ${DATA_DIR}
```

#### Running the application

Run the script run_nichart.sh that the installer created in the same directory:

```
./run_nichart.sh
```

This will start the NiChart server on your machine which you can then access from your web browser.
When you start the server, a few links will appear, including a localhost one: http://localhost:8501 

You can click that link or copy-paste it into a browser to access the local NiChart server. If you see the survey page, congratulations! NiChart is succesfully installed.

To stop the NiChart server, run "docker stop nichart_server". 

#### Updating

To update NiChart, just download and re-run the latest installer.

To save space, you may want to clean up your Docker images to remove older tool versions. For more information on managing Docker images, see the [Docker image docs](https://docs.docker.com/reference/cli/docker/image/).

# Can't use Docker?
We aim to soon provide compatibility with Singularity/Apptainer runtimes for users in computing environments where Docker is disallowed or where other related policies prevent running NiChart due to required privileges. Please check in regularly for updates.
