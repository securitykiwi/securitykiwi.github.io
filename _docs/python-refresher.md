---
title: Python Refresher
author:
---

<style>p {text-align: justify;}</style>

Python, data analysis and machine learning go hand-in-hand - Python is by far the most popular programming language for data analysis and machine learning. If you’ve never programmed in Python, or are a bit rusty, don’t worry, this section serves to get you up to speed. If you're looking for a more comprehensive set of tutorials, see the <a href="https://docs.python.org/3/tutorial/index.html" target="_blank">The Python Tutorial</a> from the <a href="https://docs.python.org/3/index.html" target="_blank">Python Documentation</a>. 

If you want to delve into machine learning once you have completed this course and don’t yet know Python well, I recommend you set aside some time to learn. If you're a fan of books <a href="https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888" target="_blank">Learn Python 3 the Hard Way</a> is an excellent book, despite the name. Much like this course, you learn through practical tutorials and examples.

Beyond a specific language, learning more about computer science will help you in any endeavour related to programming. The massively online open course (MOOC) from Harvard University, <a href="https://www.edx.org/course/cs50s-introduction-to-computer-science" target="_blank">CS50: Introduction to Computer Science</a> is highly recommended.

You can follow along with this refresher in <a href="https://colab.research.google.com/drive/1IlTHLwWdTeBXBcCqFSGN4AgJA_jvWxS_" target="_blank">Google Colab</a> live environment.

## Syntax

Python has a number of conventions laid out in PEPs, <a href="https://www.python.org/dev/peps/" target="_blank">Python Enhancement Protocols</a>, conventions shared amongst the community to encourage standardisation of behaviour. Adhering to PEPs makes code easier to read between one developer and another. Python syntax is covered by <a href="https://www.python.org/dev/peps/pep-0008/" target="_blank">PEP8</a>.

Python syntax removes anything that is considered unnecessary. As such, line separators ( ; ) are not required. Block markers ( { } ) are not required, indentation alone serves as the marker, simply indent code to create a block. A line which ends in a colon ( : ) requires an indented block to follow it, function definition lines ( def hello_world(): ) and statements which affect control flow (loops, if statements, etc) are examples. A single equals ( a = 1 ( a is 1 ) ) assigns a value to the left, double equals ( a == b ( is a equal to b? ) ) tests equality. 

Here is a small example script which prints "Hello" to a name passed in parentheses.

```python
def say_hello(name):
    print ('Hello', name)

say_hello('Alya')
say_hello('Issac')
say_hello('Mia')
```

### Comments

Single-line comments begin with a hash ( # ). Multi-line comments are surrounded by triple quotes (""").

```python
# This is a single-line comment.

"""
This is a multi-line
comment.
"""
```

Relating to comments are DocStrings, conventions for creating comment documentation for python programs. Docstrings appear at the beginning of python files and inside functions, documenting the program and individual functions. You can read more about them in <a href="https://www.python.org/dev/peps/pep-0257" target="_blank">PEP 257</a>. <a href="https://realpython.com/documenting-python-code/" target="_blank">RealPython</a> has a good article covering everything you can think of regarding python docstrings too.

### Syntax Short Hand

+= and -= can be used as shorthand to increase or decrease a value. These are referred to as subtraction assignment and addition assignment respectively. We are operating on the value and assigning the value at the same time.

In the example below, we have x with the value 10. We then subtract and assign the value 1, resulting in the value of x being 9.

```python
x = 10
x -= 1
x
```

## Strings

Strings are surrounded by either single ( ‘ ) or double ( “ ) quotation marks, it's best to stick to one type to maintain consistency. The modulo ( % ) operator and a tuple are used to insert values into a String.

The following example prints to sentences.

```python
string_one = "This is a sentence."
string_two = 'This is also a sentence'
print(string_one)
print(string_two)
```

This example accepts input, asking the user for their name and printing Hi and their name. `%s` is a <a href="https://docs.python.org/2.4/lib/typesseq-strings.html" target="_blank">string format operator</a>, a kind of placeholder, for the value after the single quotes `%name`.

```python
name = input('What is your name?\n')
print ('Hi %s.' %name)
```

## Control Flow Elements

Control flow is the order in which instructions are carried out by a computer. You can manipulate the control flow through elements such as if statements and loops. Real Python has good sections on <a href="https://realpython.com/python-conditional-statements/" target="_blank">Conditional Statements</a>, <a href="https://realpython.com/python-for-loop/" target="_blank">For Loops</a> and <a href="https://realpython.com/python-while-loop/" target="_blank">While Loops</a>.

### If Statements

An if statement allows you to branch the control flow based on a condition. The example below prints different greetings based on the name input, with the final `else` clause providing an option, should the person's name not be Mia or Issac. Note this unrealistic example is case sensitive and generally of terrible design - you would have to list all names. They would be better stored and fetched from a data structure (list, dictionary, etc).

```python
name = input('What is your name?\n')
if name = 'mia':
    print('Whatup %s.' %name)
elseif name = 'issac':
    print('Hey %s' %name)
else:
    print('Hello %s' %name)
```

### For Statement

A `for` statement allows you to loop through (iterate over) items in a data structure until a condition is met. Let us make the example above slightly less silly. Below, we ask for the user's name and check the list of four names to see if the input name is in the list. We iterate over the list and use an `if` statement to branch the flow, printing a message stating they are on the list if we find the name and a message stating they are not on the list if we don't. The break keyword breaks us out of the control flow and stops iteration.

```python
name = input('What is your name?\n')
list = ['mia', 'alya', 'issac', 'tom']
for i in list:
    if name in list:
        print('Hello %s, you are on the list!' %name)
        break
    else:
        print('Sorry, your name is not on the list.')
        break
```

### While Statement

A `while` statement allows you to iterate over items indefinitely.

```python
while True:
    do something
```

## Functions

Functions are declared with the keyword def, short for define. Arguments must be assigned a value, and values must be passed when calling arguments.

The following example takes an input word and checks to see if it is a palindrome, a word (or sentence) which is the same forwards and backwards.

```python
# Palindrome checker function
# Convert to lowercase, compare reversed word to word.
# Returns TRUE if the word is a palindrome, FALSE if not.
def palindrome_checker(word): 
  word = word.lower()
  return word == word[::-1]

# get input word
word = input("Enter a string: ") 

# Instanciate palindrome checker function
checker = palindrome_checker(word)

# See if the word is equal to its reverse (if the function returns TRUE)
if checker:
   print("It is a palindrome")
else:
   print("It is not a palindrome")
```

## Classes

Classes provide a means of bundling data and functionality together. Creating a new class creates a new type of object, allowing new instances of that type to be made. Classes support <a href="https://realpython.com/inheritance-composition-python/" target="_blank">multiple inheritance</a>.

Below is a python-pseudo-code (illustrative only) example of a class which contains information about a Student. The Student object holds the student ID, the students name and date of birth, which are initialised (<a href="https://stackoverflow.com/questions/625083/what-init-and-self-do-on-python" target="_blank">__ init __</a>) at object creation. The get_student_classes() function gets the students classes from a database.

```python
class Student:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.dob = dob
        
    def get_student_classes():
        access_student_database
        return data
```

## Exceptions

Exceptions capture errors and allow a program to fail gracefully. They are handled by try-except blocks.

This example from the python documentation shows how an exception can catch errors such as an input value not conforming to requirements (i.e. not being a number).

```python
while True:
     try:
         x = int(input("Please enter a number: "))
         break
     except ValueError:
         print("Oops!  That was no valid number.  Try again...")
```

## Data Structures: Collections

Python Collections are data structures, constructs which hold and allow the processing of data. The reasons you would choose one over another depend on the intended use, some allow multiple data types (int, float, etc) to be stored in the same structure, some do not for example. Choosing the 'right' data structure can mean less accidental mistakes by adhering to conventions and efficiency gains in some cases.

### List

Lists are ordered, changeable (mutable) and allow duplicates. Lists are denoted by square brackets. Below is an example of a list being defined and printing the last element in the list (remember we start counting at 0).

```python
list_names = ["Mia","Alya","Issac","Tom"]
print(list_names[3])
```

### Tuple

Tuples are ordered, unchangeable (immutable) and allow duplicates. Tuples are denoted by parentheses. Below we print the first element of the tuple.

```python
tuple_names = ('mia', 'alya', 'issac', 'tom')
print(tuple_names[0])
```

### Set

Sets are unordered, unindexed and do not allow duplicates. Sets are denoted by curly brackets ({}). We print all of the names in the set below.

```python
set_names = {'mia', 'alya', 'issac', 'tom'}
print(set_names)
```

### Dictionary

Dictionaries are unordered, mutable, indexed and do not allow duplicates. Dictionaries are denoted by curly brackets ({}), items consist of key and value pairs separated by a colon (:). We have to change our example slightly, due to the key-value pairs, now we are storing first and last names. We print all of the names in the Dict below.

```python
dict_names = {'Mia':'Lassila', 'Alya':'Hadid', 'Issac':'Mwangi', 'Tom':'Davies'}
print(dict_names)
```

---

## References

<ol>
    
<li>Python Software Foundation (2020) <i>The Python Tutorial</i>. <a href="https://docs.python.org/3/tutorial/index.html" target="_blank">https://docs.python.org/3/tutorial/index.html</a>
</li>

<li>Shaw, Z. (2017) <i>Learn Python 3 the Hard Way</i>. <a href="https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888" target="_blank">https://www.amazon.com/Learn-Python-Hard-Way-Introduction/dp/0134692888</a>
</li>

</ol>
