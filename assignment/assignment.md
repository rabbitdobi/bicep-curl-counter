# Assignment: Rep Counter

**Due:** before our next meeting (\_date: 2026/10/30)

Build a program that watches you do bicep curls through the webcam and  
counts them, using the elbow angle from MediaPipe Pose. Then record the  
data and plot it.

```
arm straight  →  curl up  →  back down   =  1 rep
   ~170°           ~40°        ~170°
```

This brings together most of the lessons: the camera loop (01), drawing (03),  
recording (04), pose landmarks (09), states with a dead zone (10), joint  
angles (11), and saving and plotting data (12, 13).

## What you get

`assignment/assignment_starter.py` already opens the camera, runs pose  
detection and draws your arm. It runs as it is, but it doesn't count yet.  
Search the file for **TODO**; each one has instructions in its docstring.

```bash
python assignment/assignment_starter.py
```

## Tasks

| # | Task                                                                  | Where                                 |
| - | --------------------------------------------------------------------- | ------------------------------------- |
| 1 | Compute the elbow angle                                               | `elbow_angle()`                       |
| 2 | Count reps with two thresholds (`UP_ANGLE`, `DOWN_ANGLE`)             | `RepCounter.update()`                 |
| 3 | Show the count, stage and angle clearly on screen                     | `draw_info()`                         |
| 4 | Save time, angle, stage and reps to a CSV file                        | `RepLogger.write()`                   |
| 5 | Plot your CSV: the angle over time, with a mark at each completed rep | a new file, `assignment/plot_reps.py` |

For task 5, start from lesson 13. Hint: a rep is completed where the `reps`  
column increases. `plt.axvline(time)` draws a vertical line at that time.

## What to submit

1. Your finished `assignment_starter.py` and `plot_reps.py`.
2. A **short screen recording** (20-30 s) of the counter working while you do  
   about 5 curls. On a Mac press **Cmd + Shift + 5** to record the screen.
3. The **CSV file** and the **plot (.png)** from that same set of curls.
4. A short **report** (half a page is enough, as a `.md` or `.txt` file)  
   answering the questions below.

## Questions for your report

1. Why do we use **two** angles (`UP_ANGLE` and `DOWN_ANGLE`) instead of one?  
   What happened when you tried using just one?
2. Did you use 2D or 3D angles? Why? What goes wrong if you face the camera  
   and curl your arm toward it?
3. Which values did you choose for `UP_ANGLE` and `DOWN_ANGLE`, and how did  
   you choose them? (Your plot is good evidence.)
4. Name one way someone could "cheat" your counter (get counts without real  
   curls). How could you prevent it?

## Grading

| Part                                                      | Points  |
| --------------------------------------------------------- | ------- |
| Elbow angle is correct and handles "not visible" (TODO 1) | 20      |
| Reps are counted correctly, no double counts (TODO 2)     | 25      |
| Clear, readable on-screen display (TODO 3)                | 15      |
| CSV saved correctly (TODO 4)                              | 10      |
| Plot with rep markers (task 5)                            | 15      |
| Report answers                                            | 15      |
| **Total**                                                 | **100** |

## Bonus ideas (pick any)

- Count **both arms** separately.
- Warn "**Too fast!**" if a rep takes less than one second.
- Detect a "lazy" rep: the arm never fully straightens. Don't count it.
- Show a **range-of-motion score**: smallest and largest angle in each rep.
- Use the gesture recognizer (lesson 08): a thumbs up resets the counter.

## Tips

- Stand so your **whole arm** is in view, about 2 m from the camera, side-on  
  or at an angle works better than facing the camera straight on.
- Test your counter without standing up again and again: record yourself  
  once with lesson 04, then run  
  `python assignment/assignment_starter.py --video output/<your video>.mp4 --unmirror`.  
  (`--unmirror` is needed because lesson 04 saves the mirrored picture, which  
  would swap your left and right arms. Lesson 10 shows why.)
- Print the angle to the terminal while you curl to choose good thresholds.
- If it counts twice per curl, your thresholds are too close together.
