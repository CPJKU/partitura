# Contributing

You can help the developers of **Partitura** by contributing, requesting features or reporting error.

## Opening an Issue

To open an issue navigate to the partitura github repository:

[Repository]: https://github.com/CPJKU/partitura/issues	"Partitura Issues Link"

#### Click to **New Issue**

![](docs/source/images/issue_page.png)

#### Write your description

![](docs/source/images/writing_issue.png)

#### Choose the appropriate label

![](docs/source/images/issue_choosing_label.png)

##### How to choose your issue label:

- **Question** to ask us a question or **help wanted** if you need a solution to a particular partitura problem.
- **Bug** to report something not working correctly.
- **Enhancement** to request for a feature.

## How to contribute

A step by step guide :

1. To contribute is to open a relevant issue.
2. ***Fork*** or the repo.
3. *Checkout* or *Pull* the latest stable develop branch.
4. *Checkout a new branch* from the develop with the name of your develop idea.
5. When finished coding, open a pull request.

### Open a relevant issue

Follow section how to open an issue.

## **Fork** the Repo

Fork partitura from 
https://github.com/CPJKU/partitura

Once that you have already forked the repo, you can clone it:
```shell
git clone https://github.com/YourUsername/partitura.git
cd partitura
```

### Get latest Develop Branch

```shell
git fetch *
git checkout develop
git pull
```

### Create your Branch

```shel
git checkout -b mycrazyidea
```

Do your coding magic!!!

Remember to commit regularly with descriptive messages about your changes.

**!!! IMPORTANT NOTE !!!**

Write Unit tests to check the compatibility and assure the evolution of your features.

*Please follow instruction script found in the Tutorial repository.*

### Opening your Pull Request

##### Go to Partitura Pull Requests and Click New Pull Request

[Partitura Pull Requests]: https://github.com/CPJKU/partitura/pulls	"Partitura Pull Requests"

![](docs/source/images/pull_requests.png)

##### Set the base to develop and the compare to your branch

![](docs/source/images/open_pull_request.png)

##### Then create your Pull Request and add a description.

When you create your PR then the partitura Unitests including the Unit Tests you wrote are ran.

If there is no conflict with the develop branch then you will see this on your screen :

![](docs/source/images/unitest_pass.png)

If indeed the tests pass then a person from the development team of Partitura will review your work and accept your Pull Request.

Your features will then be included to the next release of Partitura.