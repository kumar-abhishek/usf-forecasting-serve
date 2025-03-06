# usf-forecasting-serve


```
docker build --no-cache -t forecast-api .
docker run -p 8000:8000 forecast-api
```


---
log:5 march

while running this command,


i get this error:
```
another error while runnning

  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/app/app/app.py", line 16, in <module>
    model = pickle.load(f)
_pickle.UnpicklingError: could not find MARK ```

need to train model via train.py and then dump the pkl file.