import psycopg2
import numpy as np
from psycopg2 import OperationalError
from psycopg2.extensions import register_adapter, AsIs

class postgress_logger():
    def __init__(self):
        self.conn = self.make_connection()
        self.bucket_name = "data-source-brain-tumor-classification"
        self.flag = True   
        self.image_path = None
        self.validation_flag = None 
        self.correlation = None
        self.prediction = None
        self.glioma_probability = None
        self.meningioma_probability = None
        self.no_tumor_probability = None
        self.pituitary_probability = None
        self.feedback_flag = None 
        self.feedback_class = None
        self.time_taken = None 
    
    def make_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(
                host='10.73.96.6',
                port=5432,
                database='backend-logs',
                user="postgres",
                password="admin"
            )
            print("PostgreSQL connection is successful")
        except OperationalError as e:
            print(f"Error: {e}")
        finally:
            return conn
    
    @staticmethod
    def addapt_numpy_float64(numpy_float64):
        return AsIs(numpy_float64)
    
    def push_to_postgres(self):
        if not self.flag:
            # Implement your logic to push data to PostgreSQL here
            if self.conn:
                # Example: Execute a SQL query to insert data
                try:
                    cursor = self.conn.cursor()
                    # Example query (replace with your actual insert query)
                    query = """
                        INSERT INTO public.backendlogs(
                            imagepath_url,
                            validation_flag,
                            prediction,
                            glioma_probability_1,
                            meningioma_probability_2,
                            no_tumor_probability_3,
                            pituitary_probability,
                            time_taken,
                            correlation        
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    data = (self.image_path,self.validation_flag,self.prediction,AsIs(self.glioma_probability),AsIs(self.meningioma_probability),AsIs(self.no_tumor_probability),AsIs(self.pituitary_probability), AsIs(self.time_taken),AsIs(self.correlation))
                    print(data)
                    cursor.execute(query, data)  # Execute the insert statement
                    self.conn.commit()
                    print("Data pushed to PostgreSQL")
                except Exception as e:
                    print(f"Error pushing data to PostgreSQL: {e}")
                finally:
                    cursor.close()
        else:
            print("Flag is True, skipping push to PostgreSQL")
            
    def push_feedback_postgres(self):
        if self.feedback_flag:
            # Implement your logic to push data to PostgreSQL here
            if self.conn:
                # Example: Execute a SQL query to insert data
                try:
                    cursor = self.conn.cursor()
                    # Example query (replace with your actual insert query)
                    query = """
                        UPDATE public.backendlogs
                        SET
                            feedback_flag = %s,
                            feedback_class = %s
                        WHERE
                            imagepath_url = %s;
                    """
                    data = (self.feedback_flag, self.feedback_class, self.image_path)
                    print(data)
                    cursor.execute(query, data)
                    self.conn.commit()
                    print("Data pushed to PostgreSQL")
                except Exception as e:
                    print(f"Error pushing data to PostgreSQL: {e}")
                finally:
                    cursor.close()
        else:
            print("Flag is True, skipping push to PostgreSQL")