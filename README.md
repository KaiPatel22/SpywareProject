# GUIDE 

Ensure you set up a virtual environment ` python -m venv myenv` and then run `pip install -r requirements.txt` once inside the virtual environment.

Packets `.pcap` can be extracted with the `extractFromPcap.sh` and the activity.txt file can be converted into a python object file using `extractActivityLabels.sh`

Once the extracted packet header csv (rawPackets.csv) and python object file (activityObject.py) have been extracted the steps following are:

## Window Analysis

Run 
`
python code/windowAnalysis.py data/rawPackets.csv notes/activityObject.py --flag
`

The flag can be 

```
--sliding 
--eventCentered
--eventCenteredWithIdle
--fullEvent
--fullEventWithIdle
```

## Machine Learning

The machine learning can be run with 
`
python code/modelCreation.py data/fullEvent.csv
`

Models can be commented and uncommented to test different models. 
Models are saved in the `models/` folder