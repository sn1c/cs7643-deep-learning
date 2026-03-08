I want you to help me run experiments and optimize hyperparameters by hand editing the file `configs/config_exp.yaml`.

The only hyperparameters I want you to change is `learning_rate` (learning rate), `reg` (regularization alpha), `epochs` (epochs), and `hidden_size` (hidden size).

Please use the following grid:
- `learning_rate`: `[2, 1, 0.1, 0.05, 0.01]`
- `reg`: `[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]`
- `epochs`: `[10, 20, 30, 50]`
- `hidden_size`: `[32, 64, 128, 256]`

You don't need to try all of them, and you can make educated guesses based on previous performance.

You can run experiments using `python main.py --config configs/config_exp.yaml | grep -E  "Accuracy(.*):" | tail -n3`.

For each run, do the following:
- Each run will generate a file called `acc.png` that is a plot of the accuracy as well as a plot of learning rate across epochs called `loss.png`. Please move them in a separate folder and make them uniquely identified by 
- Only modification allowed: You can also modify `utils.py` in `plot_curves` to dump the `train_loss_history`, `train_acc_history`, `valid_loss_history`, `valid_acc_history` into a csv and use it to make educated guesses regarding over/underfitting.

After you are done, give me the following:
```
You are now free to tune any hyper-parameters for better accuracy. Create a table below and put the configuration of your best model and train/val/test accuracy into the table:

Explain why your choice works: Explanation should go into WHY things work the way they do in the context of Machine Learning theory/intuition, along with justification for your experimentation methodology. DO NOT just describe the results, you should explain the reasoning behind your choices and what behavior you expected. Also, be cognizant of the best way to mindful and show the results that best emphasizes your key observations. If you need more than one slide to answer the question, you are free to create new slides.
```