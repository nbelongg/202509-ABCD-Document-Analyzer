import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from dotenv import load_dotenv
from logger import api_logger as logger

load_dotenv(override=True)

sender_email = os.getenv("sender_email")
sender_password = os.getenv("sender_password")


class EmailService:
    def __init__(self):
        # Read sender details from environment variables
        self.sender_email = sender_email
        self.sender_password = sender_password

        # logger.info statements to debug environment variables
        logger.info(f"Sender Email: {self.sender_email}")
        logger.info(f"Sender Password: {'***' if self.sender_password else None}")

        # Check if the environment variables are set
        if not self.sender_email or not self.sender_password:
            raise ValueError("SENDER_EMAIL and SENDER_PASSWORD environment variables must be set.")

        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        logger.info("EmailService initialized with SMTP server and port.")

    def send_email(self, recipient_email, attachments=None):
        logger.info(f"Preparing to send email to: {recipient_email}")
        logger.info(f"Attachments: {attachments}")

        # Hardcoded subject and body
        subject = "PDF Analysis Results"
        body = "Attached is the CSV file containing the analysis results of your PDF documents."

        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Add body to email
        msg.attach(MIMEText(body, "plain"))
        logger.info("Email body attached.")

        # Add attachments if any
        if attachments:
            try:
                with open(attachments, "rb") as f:
                    attachment = MIMEApplication(f.read(), _subtype="txt")
                    attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachments))
                    msg.attach(attachment)
                logger.info(f"Attachment {attachments} added.")
            except Exception as e:
                logger.info(f"Failed to attach file {attachments}: {e}")

        try:
            # Create SMTP session
            logger.info("Creating SMTP session...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable security
            logger.info("SMTP session started with TLS.")

            # Login to the server
            logger.info("Logging in to SMTP server...")
            server.login(self.sender_email, self.sender_password)
            logger.info("Logged in successfully.")

            # Send email
            logger.info("Sending email...")
            server.send_message(msg)
            logger.info("Email sent successfully!")

        except Exception as e:
            logger.info(f"An error occurred: {e}")

        finally:
            logger.info("Closing SMTP session.")
            server.quit()

    def send_failure_notification(self, recipient_email, batch_id, status):
        logger.info(f"Preparing to send failure notification email to: {recipient_email}")

        # Create email subject and body for failure notification
        subject = f"PDF Analysis Batch {batch_id} Failed"
        body = f"""
        Your PDF analysis batch request has failed.
        
        Batch ID: {batch_id}
        Status: {status}
        
        Please contact support if you need assistance.
        """

        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Add body to email
        msg.attach(MIMEText(body, "plain"))
        logger.info("Failure notification email body attached.")

        try:
            # Create SMTP session
            logger.info("Creating SMTP session...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable security
            logger.info("SMTP session started with TLS.")

            # Login to the server
            logger.info("Logging in to SMTP server...")
            server.login(self.sender_email, self.sender_password)
            logger.info("Logged in successfully.")

            # Send email
            logger.info("Sending failure notification email...")
            server.send_message(msg)
            logger.info("Failure notification email sent successfully!")

        except Exception as e:
            logger.info(f"An error occurred while sending failure notification: {e}")

        finally:
            logger.info("Closing SMTP session.")
            server.quit()


# Example usage
if __name__ == "__main__":
    email_service = EmailService()
    recipient_email = "manishsparihar2020@gmail.com"
    subject = "Test Email from Python"
    body = "This is a test email sent from Python!"

    # Optional: Add attachments
    # attachments = ["path/to/file.txt"]  # Add file paths here

    # Send email without attachments
    email_service.send_email(recipient_email)

    # Send email with attachments
    # email_service.send_email(recipient_email, subject, body, attachments)
