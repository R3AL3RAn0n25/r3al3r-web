# R3AL3R AI User Manual

## Overview
R3AL3R AI is the premier AI companion app, offering personalized insights, a market simulation game, and the ability to ask about any topic. All responses are ethically reviewed to ensure alignment with the highest standards, making R3AL3R AI secure and trustworthy.

## Installation Instructions

### Prerequisites
- **iOS**: iOS 13.0+, 2GB RAM, 200MB storage.
- **Android**: Android 8.0+, 2GB RAM, 200MB storage.
- **Web**: Modern browser (Chrome, Firefox, Safari), WebAuthn support.
- **Backend** (for developers):
  - Python 3.9+, MongoDB, Redis, Flask.
  - SSL certificates for HTTPS.
  - Fail2Ban for intrusion detection.

### iOS Installation
1. **Download**:
   - Visit `https://api.r3al3r.ai/download/ios` to download the beta `.ipa` file (available via TestFlight).
2. **Install**:
   - Open TestFlight, add the app, and install.
3. **Permissions**:
   - Grant microphone access for voice queries.
   - Allow notifications for ethical review alerts (admin users only).
4. **Verify**:
   ```bash
   xcrun simctl install booted r3al3r_app.ipa  # For simulator testing
   ```

### Android Installation
1. **Download**:
   - Visit `https://api.r3al3r.ai/download/android` to download the `.apk` file.
2. **Install**:
   - Enable "Install from unknown sources" in Settings.
   - Install the `.apk` file.
3. **Permissions**:
   - Grant microphone and notification permissions.
4. **Verify**:
   ```bash
   adb install app-release.apk  # For emulator testing
   ```

### Web Installation
1. **Access**:
   - Open `https://app.r3al3r.ai` in a WebAuthn-compatible browser.
2. **Login**:
   - Use WebAuthn for secure authentication (biometric or security key).
3. **Verify**:
   - Ensure WebAuthn is enabled in browser settings.

### Backend Setup (Developers)
1. **Clone Repository**:
   ```bash
   git clone https://github.com/r3al3r/r3al3r-ai.git
   cd r3al3r-ai
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure MongoDB**:
   ```bash
   mongod --config mongod.conf
   mongo admin --eval 'db.createUser({user:"admin",pwd:"secure_password",roles:["root"]})'
   ```
4. **Configure Redis**:
   ```bash
   redis-server --requirepass secure_redis_password
   ```
5. **Generate SSL Certificates**:
   ```bash
   certbot certonly --standalone -d api.r3al3r.ai
   cp /etc/letsencrypt/live/api.r3al3r.ai/fullchain.pem cert.pem
   cp /etc/letsencrypt/live/api.r3al3r.ai/privkey.pem key.pem
   ```
6. **Setup Fail2Ban**:
   ```bash
   cp jail.local /etc/fail2ban/jail.local
   cp r3al3r-api.conf /etc/fail2ban/filter.d/r3al3r-api.conf
   systemctl restart fail2ban
   ```
7. **Run Backend**:
   ```bash
   export SOUL_KEY_HASH=$(echo -n "your_soul_key" | sha256sum | awk '{print $1}')
   python r3al3r_ai_framework.py
   ```

## Running the Application

### iOS/Android
1. **Launch**:
   - Open the R3AL3R AI app.
2. **Login**:
   - Enter your User ID and Soul Key.
   - For web, use WebAuthn (biometric or security key).
3. **Dashboard**:
   - View insights, global impact ideas, and personalized recommendations.
   - Tap "Ask Anything" to query any topic.
   - Tap "Play Market Game" for AI-driven challenges.
   - Use "Speak" for voice queries.
4. **Ethical Review (Admin)**:
   - Access the Ethical Review screen to approve/reject responses.
   - Receive notifications for pending reviews.

### Web
1. **Access**:
   - Navigate to `https://app.r3al3r.ai`.
2. **Login**:
   - Authenticate via WebAuthn.
3. **Usage**:
   - Same features as mobile, but voice queries may depend on browser support.

## Security Features
- **Authentication**: Secure JWT with token refresh and WebAuthn for web.
- **Encryption**: AES-256 for user data, HTTPS for all API calls.
- **Data Storage**: MongoDB with authentication, RBAC, and encryption at rest.
- **Intrusion Detection**: ML-based anomaly detection and Fail2Ban for brute-force protection.
- **Failsafes**: Global kill switch to disable app if compromised.
- **Audit Logging**: Tracks all sensitive operations in MongoDB.

## Troubleshooting
- **Login Failure**:
  - Verify User ID and Soul Key.
  - Check internet connection.
  - For web, ensure WebAuthn is enabled.
- **Query Delays**:
  - Responses may be pending ethical review.
  - Check notification for approval status (admin users).
- **Backend Errors**:
  - Verify MongoDB/Redis are running:
    ```bash
    systemctl status mongod redis
    ```
  - Check logs: `tail -f r3al3r.log`.
- **Security Alerts**:
  - If kill switch is activated, contact admin to reset.
  - Review audit logs: `mongo r3al3r_db --eval 'db.audit_logs.find().pretty()'`.

## Beta Testing Feedback
- Submit feedback via the app’s Query screen (rate 1-5).
- Report issues to `support@r3al3r.ai`.
- Suggest new knowledge sources via the app or email.

## Contact
For support, contact `support@r3al3r.ai` or visit `https://r3al3r.ai/support`.