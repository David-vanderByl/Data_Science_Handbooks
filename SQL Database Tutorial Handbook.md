# SQL Database Tutorial Handbook

**Note that this tutorial is largely untested!**

## Table of Contents
1. [Introduction to SQL Databases](#chapter-1-introduction-to-sql-databases)
   - [Overview](#11-overview)
   - [SQL Queries with Comprehensive Examples](#12-sql-queries-with-comprehensive-examples)

<br>

2. [Setting up a Toy Database with Docker](#chapter-2-setting-up-a-toy-database-with-docker)
   - [Using Docker for MySQL](#21-using-docker-for-mysql)
   - [Employing the Example Database](#22-employing-the-example-database)

<br>


3. [Using a DBMS to Pull Data](#chapter-3-using-a-dbms-to-pull-data)
   - [Terminal Instructions for Data Extraction](#31-terminal-instructions-for-data-extraction)
   - [Using DBeaver for Database Management](#32-using-dbeaver-for-database-management)

<br>


4. [Querying Data via Python API](#chapter-4-querying-data-via-python-api)
   - [Introduction to Python MySQL Library](#41-introduction-to-python-mysql-library)
   - [Querying Data with Python](#42-querying-data-with-python)

<br>


5. [Querying with Pandas `read_sql`](#chapter-5-querying-with-pandas-read-sql)
   - [Introduction to Pandas and SQLAlchemy](#51-introduction-to-pandas-and-sqlalchemy)
   - [Querying Data with Pandas](#52-querying-data-with-pandas)

<br>


6. [Writing Data with Pandas `to_sql`](#chapter-6-writing-data-with-pandas-to-sql)
   - [Writing Data to the Database](#61-writing-data-to-the-database)
   - [Managing Indexes and `if_exists` Parameter](#62-managing-indexes-and-if_exists-parameter)

<br>


7. [Advanced SQL Concepts](#chapter-7-advanced-sql-concepts)
   - [Subqueries and Derived Tables](#71-subqueries-and-derived-tables)
   - [Window Functions](#72-window-functions)
   - [Indexes and Optimization](#73-indexes-and-optimization)

<br>


8. [Transactions and ACID Properties](#chapter-8-transactions-and-acid-properties)

<br>


9. [Data Modeling and Normalization](#chapter-9-data-modeling-and-normalization)
   - [Normal Forms](#91-normal-forms)
   - [Example of Normalization](#92-example-of-normalization)

<br>


10. [Backups, Restores, and Security](#chapter-10-backups-restores-and-security)
   - [Backing Up Data](#101-backing-up-data)
   - [Restoring Data](#102-restoring-data)
   - [Basic Security Practices](#103-basic-security-practices)

<br>


11. [Conclusion](#conclusion)





Feel free to use these links to navigate directly to each chapter and section.

### Chapter 1: Introduction to SQL Databases
In this chapter, we will provide an overview of SQL databases, their importance, and the topics covered in the tutorial. We will also delve into SQL queries and provide comprehensive examples to cover the fundamental aspects of querying data.

#### 1.1 Overview
SQL (Structured Query Language) databases are a crucial part of modern software applications for storing, managing, and retrieving structured data. They provide a structured way to interact with data and are widely used across industries. This tutorial will guide you through the process of working with SQL databases using Python, focusing on MySQL.

#### 1.2 SQL Queries with Comprehensive Examples
SQL queries are the backbone of interacting with databases. Let's cover some fundamental SQL query types along with examples:

**SELECT Statement:**
Retrieve specific columns from a table.
```sql
SELECT first_name, last_name FROM employees;
```

**WHERE Clause:**
Filter data based on conditions.
```sql
SELECT * FROM employees WHERE department = 'Sales';
```

**JOIN Operation:**
Combine data from multiple tables.
```sql
SELECT employees.first_name, departments.department_name
FROM employees
INNER JOIN departments ON employees.department_id = departments.department_id;
```

**Aggregate Functions:**
Perform calculations on data.
```sql
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;
```

**Subqueries:**
Use a query inside another query.
```sql
SELECT first_name, last_name
FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE department_name = 'HR');
```

### Chapter 2: Setting up a Toy Database with Docker
In this chapter, we will guide you through the process of setting up a toy database using Docker and MySQL. We will utilize the example database provided by MySQL for practice.

#### 2.1 Using Docker for MySQL
Docker allows us to run applications in isolated containers. Here's how to set up MySQL using Docker:

1. Install Docker on your platform.
2. Pull the MySQL image:
   ```bash
   docker pull mysql:latest
   ```
3. Run a MySQL container:
   ```bash
   docker run --name=mysql-container -e MYSQL_ROOT_PASSWORD=password -d -p 3306:3306 mysql:latest
   ```

#### 2.2 Employing the Example Database
MySQL provides an example database that simulates employee data. Let's set it up:

1. Download the example database from [here](https://dev.mysql.com/doc/employee/en/employees-validation.html).
2. Follow the provided instructions to populate the database using SQL scripts.

### Chapter 3: Using a DBMS to Pull Data
This chapter focuses on using a Database Management System (DBMS) to extract data from the MySQL database. We'll provide terminal instructions and demonstrate using DBeaver, a popular database management tool.

#### 3.1 Terminal Instructions for Data Extraction
Assuming you have the MySQL command-line client installed, you can interact with the database using terminal commands:

1. Connect to the MySQL server:
   ```bash
   mysql -h localhost -u root -p
   ```
2. Enter your password and start querying:
   ```sql
   SELECT * FROM employees;
   ```

#### 3.2 Using DBeaver for Database Management
DBeaver is a powerful tool for managing databases visually. Here's how to set it up:

1. Install DBeaver on your platform.
2. Open DBeaver and create a new database connection.
3. Choose MySQL as the database type and provide connection details.
4. Once connected, you can execute SQL queries using the built-in SQL editor.

### Chapter 4: Querying Data via Python API
In this chapter, we will explore how to query data from the MySQL database using Python's MySQL library. We'll work within a Jupyter notebook environment to demonstrate the process.

#### 4.1 Introduction to Python MySQL Library
Python provides various libraries for interacting with databases. Install the MySQL library using:
```bash
pip install mysql-connector-python
```

#### 4.2 Querying Data with Python
Let's use the MySQL library to query data within a Jupyter notebook:

```python
import mysql.connector

# Establish a connection
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="employees"
)

# Create a cursor
cursor = connection.cursor()

# Execute a query
query = "SELECT * FROM employees"
cursor.execute(query)

# Fetch and display results
results = cursor.fetchall()
for row in results:
    print(row)

# Close cursor and connection
cursor.close()
connection.close()
```

### Chapter 5: Querying with Pandas `read_sql`
In this chapter, we will introduce using `pandas` along with SQLAlchemy to query data directly from the MySQL database. This approach simplifies data handling and analysis.

#### 5.1 Introduction to Pandas and SQLAlchemy
`pandas` is a popular data manipulation library, and SQLAlchemy is a powerful toolkit for database interaction. Install the required libraries using:
```bash
pip install pandas sqlalchemy
```

#### 5.2 Querying Data with Pandas
```python
import pandas as pd
from sqlalchemy import create_engine

# Create a database connection using SQLAlchemy
engine = create_engine("mysql+mysqlconnector://root:password@localhost/employees")

# Query data into a DataFrame
query = "SELECT * FROM employees"
df = pd.read_sql(query, engine)

# Perform data analysis and manipulation on the DataFrame
print(df.head())
```

### Chapter 6: Writing Data with Pandas `to_sql`
This chapter focuses on writing data back to the MySQL database using `pandas`'s `to_sql()` function. We will explore various scenarios, including handling duplicates and managing indexes.

#### 6.1 Writing Data to the Database
```python
# Assume 'new_data' is a DataFrame with new records
new_data.to_sql(name='employees', con=engine, if_exists='append', index=False)
```

#### 6.2 Managing Indexes and `if_exists` Parameter
```python
# Replace existing table with new_data and create indexes
new_data.to_sql(name='employees', con=engine, if_exists='replace', index=True)
```


### Chapter 7: Advanced SQL Concepts

#### 7.1 Subqueries and Derived Tables
Subqueries allow you to perform complex operations by nesting one query within another. Derived tables create a temporary table for use in your main query.

**Subquery Example:**
```sql
SELECT first_name, last_name
FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE department_name = 'HR');
```

**Derived Table Example:**
```sql
SELECT employees.first_name, employees.last_name, department_totals.total_salary
FROM employees
JOIN (SELECT department_id, SUM(salary) AS total_salary FROM employees GROUP BY department_id) AS department_totals
ON employees.department_id = department_totals.department_id;
```

#### 7.2 Window Functions
Window functions compute values across a set of table rows related to the current row. They allow for advanced analytical operations without the need for self-joins or subqueries.

```sql
SELECT first_name, last_name, salary, 
       AVG(salary) OVER (PARTITION BY department_id) AS avg_department_salary
FROM employees;
```

#### 7.3 Indexes and Optimization
Indexes are crucial for query performance. They allow the database to quickly locate the rows that match a query. Properly designed indexes significantly improve search and retrieval times.

**Creating an Index:**
```sql
CREATE INDEX idx_department ON employees (department_id);
```

**Analyzing Query Execution:**
```sql
EXPLAIN SELECT * FROM employees WHERE department_id = 2;
```

### Chapter 8: Transactions and ACID Properties

Transactions ensure data integrity and follow the ACID properties: Atomicity, Consistency, Isolation, and Durability.

```sql
-- Beginning a Transaction
START TRANSACTION;

-- Performing Multiple SQL Operations within the Transaction
UPDATE accounts SET balance = balance - 500 WHERE account_id = 123;
UPDATE accounts SET balance = balance + 500 WHERE account_id = 456;

-- Committing the Transaction
COMMIT;

-- Rolling Back the Transaction
ROLLBACK;
```

### Chapter 9: Data Modeling and Normalization

Data modeling involves designing a database structure for efficient storage and retrieval. Normalization eliminates data redundancy and ensures consistency.

#### 9.1 Normal Forms
- **First Normal Form (1NF):** Eliminate repeating groups and ensure atomicity.
- **Second Normal Form (2NF):** Meet 1NF and remove partial dependencies.
- **Third Normal Form (3NF):** Meet 2NF and eliminate transitive dependencies.

#### 9.2 Example of Normalization
Consider an example where we have employee data and departments. A denormalized table might store both employee and department information. However, a normalized approach separates these entities into different tables to reduce redundancy.

### Chapter 10: Backups, Restores, and Security

Protecting your data is paramount. This chapter covers data backup, restoration, and basic security practices.

#### 10.1 Backing Up Data
Regular backups prevent data loss. Use tools like `mysqldump` to create backups.

```sql
-- Creating a Backup Using mysqldump
mysqldump -u username -p database_name > backup.sql
```

#### 10.2 Restoring Data
Restore data from a backup to recover lost information.

```sql
-- Restoring a Backup Using MySQL Client
mysql -u username -p database_name < backup.sql
```

#### 10.3 Basic Security Practices
Ensure database security with strong passwords, limited user privileges, and regular security updates.

### Conclusion

With your completion of this comprehensive SQL database tutorial, you now possess a solid foundation for working with SQL databases in Python. From basic queries to advanced concepts, you're equipped to handle various database-related tasks. Continue to practice and explore real-world scenarios to further develop your skills. Happy coding and database management!