# Installing NiChart

We provide both a locally installable **desktop application** and a **cloud-based application**.

The [NiChart cloud application](https://neuroimagingchart.com/portal), hosted via Amazon Web Services (AWS), deploys scalable infrastructure which hosts the *NiChart* tools as a standard web application accessible via the user’s web browser. **No install needed**, but it requires you to upload your data to the cloud-based NiChart server for us to process it. We do not access or use your data for any other purpose than to run your requested processing and/or provide support to you, and we regularly delete user data after inactivity. However, we recognize that data privacy agreements and related concerns may nevertheless restrict use of the cloud application. If that applies to you, we suggest that you install the desktop application. Below we provide detailed installation instructions.

In particular, if you don't have a GPU on your device, the cloud application is probably the easiest way for you to use the NiChart tools.

The cloud and desktop applications are unified at the code level through the use of the Python library [Streamlit](https://streamlit.io). Consequently, the user experience is nearly identical between the cloud and desktop applications.

**Desktop installation**: Installing the desktop application currently requires [Docker](https://www.docker.com/get-started/) to be installed, as this greatly simplifies deployment and distribution of our algorithms without requiring extensive dependency management. Follow the instructions to install Docker (or Docker Desktop, on Windows/Mac) for your platform, then restart your device before continuing. We recommend having at least 20 GB of free space on your device before installing NiChart.


## Windows Instructions

Windows users will likely need to first [install the Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). 

On Windows, Docker is distributed as "Docker Desktop", an application which manages Docker on your system. 

### Docker-based Installation

#### Getting started

First, if you're on Windows, open Docker Desktop. You can do this from the start/search menu or by clicking the Desktop shortcut if you selected that during installation. You should go into the settings using the gear icon on the top right, go to "General", and enable the settings "Use the WSL 2 based engine" and "Expose daemon on tcp://localhost:2375 without TLS" if they aren't already enabled (they might require you to restart). You should also see a green indicator on the bottom left which says "Engine running". If it's yellow, you need to wait for the service to start. Otherwise, you may need to troubleshoot your installation. 

#### Choose a path to store results

In this installation, NiChart runs inside a container, which is isolated from the rest of your computer to improve security. To have data persist across sessions, you need to designate a location on your computer to store this data. ****

First, identify a path you want to use. In this demo we'll use "C:/Users/NiChart_Developer/Desktop/DEMODATA", but yours will vary as you can choose any folder you like. On Windows, you can navigate to a folder, then click "copy path" in the file explorer to get your path.

**Please make sure that the path you use does not already contain important data**. NiChart will not try to delete existing data, but it is good practice to select a new, empty folder.

Write down your path (for example, copy & paste it into Notepad).

Now, in your path text, replace "C:/"with "/mnt/c/". You can do the same for any other drive letter, so "D:/" becomes "/mnt/d". In our example, we end up with "/mnt/c/Users/NiChart_Developer/Desktop/DEMODATA". 

Write down this converted path as we will use it later. 

#### Running the installer

Make sure you are connected to the internet in order to download the application. Then, open a terminal.

(On Windows, search "terminal", open the application that looks like a black box with a white ">_" in it. At the top of the window that appears will be a tab indicating Windows Powershell.
Click the down arrow next to that tab to expand your terminal options, and select Ubuntu (or, if you changed the default distribution, select whichever distribution you selected while installing WSL).
A new terminal will open in a different color and you should see something like "root@username:~#". Stay on this tab for the rest of the instructions.)

Run the following commands to download the installer and make it runnable:

```
wget https://raw.githubusercontent.com/CBICA/NiChart_Project/main/installers/install_nichart_docker_linux.sh.sh
chmod +x install_nichart_docker_linux.sh
```

Then run this command, **making sure to replace** /path/to/data with the data path you chose earlier:
```
./install_nichart_docker_linux.sh /path/to/data
```

In our example, the command becomes `./install_nichart_docker_linux.sh /mnt/c/Users/NiChart_Developer/Desktop/DEMODATA`.

This command might take a while to finish.

#### Running the application

To run the application, run the following command.

```
./run_nichart.sh
```

This will start the NiChart server on your machine which you can then access from your web browser.
When you start the server, a few links will appear, including a localhost one: http://localhost:8501 

You can click that link or copy-paste it into a browser to access the local NiChart server. 

The NiChart server will automatically stop when you close that terminal window.

Whenever you want to run NiChart again, open up the Ubuntu terminal as described above and run the same command. Then open your browser and go to http://localhost:8501 



## Linux Instructions
Linux instructions are quite similar to the Windows instructions, except that there is no need for WSL and no need to convert the path specially. All commands will need to be run from the terminal and there is no Docker Desktop configuration.

# Can't use Docker?
We aim to soon provide compatibility with Singularity/Apptainer runtimes for users in computing environments where Docker is disallowed. Please check in regularly for updates.
