use employees;
show tables;

SELECT *
FROM employees
JOIN dept_emp USING (emp_no)
JOIN departments USING (dept_no)
;

SELECT * 
FROM employees;

SELECT de.emp_no, e.gender, de.dept_no, d.dept_name
FROM dept_emp as de
JOIN departments as d USING (dept_no)
JOIN employees as e USING (emp_no)
WHERE de.to_date > NOW()
ORDER BY de.emp_no
;

SELECT *
FROM dept_manager as de
JOIN employees as e USING (emp_no)
JOIN departments as d USING (dept_no)
;

SELECT COUNT(*)
FROM dept_emp as de
JOIN employees as e USING (emp_no)
;

SELECT emp_no,
	IF (True, True, False) as is_manager
FROM dept_manager;

SELECT emp_no, gender, is_manager
FROM employees
LEFT JOIN (
	SELECT emp_no,
		IF (True, True, False) as is_manager
	FROM dept_manager
	) as dm USING (emp_no)
;