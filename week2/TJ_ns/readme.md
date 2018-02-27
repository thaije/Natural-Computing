# Negative selection

## Exercise 1

In this exercise we generate a repertoire of string-based T cells, trained on
the English book Moby Dick, to recognize other languages.

The training and test sets contain strings of 10 characters long, with spaces replaced for
underscores etc.
Some examples:

```
call_me_is
shore_i_th
my_purse_a
```

### Generation of the T cells
We generate a repertoire of T-cells, which are also 10-char long strings.
They are randomly generated.
For example:

```
ras_dsl_as
oiasdi_oai
tle_and_se
```


### Training of the T cells

After generating of the T cells, we have to remove the T cells which attack the
bodies own cells, or in this case match the strings in the training file.

As an example, the generated T cell string `tle_and_se` is also present in the
training file. Thus this one is removed.

A T cell needs to be removed if it contains a contingouos 4-char substring
which is also present in a training string, and thus is too similar to the training string.

All generated T cells are tested against all training strings, which will only leave
T cells which react to anomalies, strings not in the training set.
Hopefully this will make the model able to distinguish between English on which it was trained, and other languages.  



### Commands
Test on Tagalog test set `java -jar negsel2.jar -self english.train -n 10 -r 4 -c -l < tagalog.test | awk '{n+=$1}
END{print n/NR}'`
Test on English test set `java -jar negsel2.jar -self english.train -n 10 -r 4 -c -l < english.test | awk '{n+=$1}
END{print n/NR}'' `
