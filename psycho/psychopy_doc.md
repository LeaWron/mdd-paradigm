# session

A Session is from which you can run multiple PsychoPy experiments, so long as they are stored within the same folder. Session uses a persistent Window and inputs across experiments, meaning that you don’t have to keep closing and reopening windows to run multiple experiments.

Through the use of multithreading, an experiment running via a Session can be sent commands and have variables changed while running. Methods of Session can be called from a second thread, meaning they don’t have to wait for runExperiment to return on the main thread. For example, you could pause an experiment after 10s like so:

```python
# define a function to run in a second thread 
def stopAfter10s(thisSession):
    # wait 10s 
    time.sleep(10) 
    # pause 
    thisSession.pauseExperiment()

# create a second thread 
thread = threading.Thread(target=stopAfter10s, args=(thisSession,)) 

# start the second thread 
thread.start() 

# run the experiment (in main thread) 

thisSession.runExperiment(“testExperiment”)
```

When calling methods of Session which have the parameter blocking from outside of the main thread, you can use blocking=False to force them to return immediately and, instead of executing, add themselves to a queue to be executed in the main thread by a while loop within the start function. This is important for methods like runExperiment or setupWindowFromParams which use OpenGL and so need to be run in the main thread. For example, you could alternatively run the code above like this:

```python
 # start the experiment in the main thread 
 thisSession.runExperiment(“testExperiment”, blocking=False) 
 # wait 10s 
 time.sleep(10) 
 # pause 
 thisSession.pauseExperiment()
 
 # create a second thread 
 thread = threading.Thread(target=stopAfter10s, args=(thisSession,))
 
 # start the second thread 
 thread.start() 
 
 # start the Session so that non-blocking methods are executed 
 thisSession.start()
```

## 未说明的部分

要使用 py 脚本作为被管理的实验，脚本中需要实现这些接口

```python
expInfo = {} # 实验的信息，包括名称、session id，受试信息等，可以通过 DlgFromDict 修改

def showExpInfoDlg(expInfo) -> dict[str, Any]: # 展示 gui 进行修改等
    return expInfo

def setupData(expInfo) -> ExperimentHandler:
    pass

def setupLogging() -> LogFile: # 可选
    pass

def setupWindow(expInfo) -> Window 
    pass

def setupInputs(expInfo, thisExp: ExperimentHandler, win:Windows) -> dict[str, Any]:
    pass

def pauseExperiment(thisExp, inputs, win, timers=[], playbackComponets=[]): # 供 session 暂停使用
    pass

def run(expInfo, thisExp, win, inputs): # 开启实验的接口
    pass

def endExperiment():
    pass

def saveData(thisExp):
    pass

def quit(thisExp, win, inputs):
    pass


"""
一般的调用顺序如下

if __name__ == "__main__":
    expInfo = showExpInfoDlg(expInfo)
    thisExp = setupData(expInfo)
    logFile = setupLogging(thisExp.dataFileName)
    win = setupWindow(expInfo)
    inputs = setupInputs(expInfo, thisExp, win)
    run(expInfo, thisExp, win, inputs)
    saveData(thisExp)
    quit(thisExp, win, inputs)
"""
```

更多信息可以在 examples 目录下的 [example_go-nogo.py](../examples/example_go-nogo.py)
