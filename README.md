# Testing and Improving the Generalisation of Probe-Based Monitors
https://docs.google.com/document/d/182KWdsFIF3AzPo8EDRnpG_-8IV5zWDSchSA2aNO95aE/edit?tab=t.0#heading=h.9lmc73wscx1r

- For each research direction, create a new directory in the research folder
- Add generic reusable code in the experiment_library folder, especially for larger scale experiments
- If you don't want to use the library, copy the notebooks and scripts from experiment_standalone_templates, especially for quick experiments
	- 'FullTemplate' generates a single dataset, then trains and evaluates the probe on that dataset
	- 'FullTemplateWithOOD' generates an ID and OOD dataset, then trains on ID and tests on OOD datasets
	- 'ProbeOnlyTemplate' uses existing activation datasets, then trains and evaluates the probe on these (either ID or OOD)


# Using hired GPUs
## Connect vscode to vast.ai
Open a terminal that isnt WSL (e.g. Windows/ Mac/ Native Linux) and run this command while just pressing enter for each option, to set up a private and public ssh key:
```
ssh-keygen -t rsa
```
Then copy the contents of the public key file (e.g. at "C:\Users\<username_here>\.ssh\id_rsa.pub") and add it to vast.ai at https://cloud.vast.ai/manage-keys/ clicking 'SSH Keys' tab at the top.\
Then create a GPU instance and click on 'Terminal Connection Options' near the 'Open' button to get the ssh command, which you then add to to specify your private ssh key location to. It should look like this:
```
 ssh -p 55327 root@199.126.134.31 -L 8080:localhost:8080 -i ~\.ssh\id_rsa
 ```
Then open vscode without WSL connection and select 'Connect to host...' and paste the ssh command in. \
It might ask you to choose a config to save the ssh instruction to, in which case do that and then redo 'Connect to host...' but this time just select the ssh IP from the list instead of pasting the command. \
You should be connected. Now to open the workspace folder, click File → Open folder instead of using the explorer menu. \
To move files over, you can git clone or pull from google drive or scp from local files to the instance, for example:
```
scp -P 55327 local_file.py root@199.126.134.31:/workspace
```

## Run notebooks in vscode connected to vast.ai

When not running individual python scripts and want to use Jupyter notebooks, need to set up a new kernel in the terminal:
```
uv sync && uv run python -m ipykernel install --user --name=uv-env --display-name "Python (uv)"
```
It might ask you to install vscode extensions for python and jupyter first. 
Then, click 'Select kernel' in the top right. You may need to press the refresh button if it is not showing up.