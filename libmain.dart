import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:encrypt/encrypt.dart' as encrypt;
import 'package:jwt_decoder/jwt_decoder.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:flutter_tailwindcss/flutter_tailwindcss.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:crypto/crypto.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _initNotifications();
  runApp(R3AL3RApp());
}

FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();

Future<void> _initNotifications() async {
  const AndroidInitializationSettings initializationSettingsAndroid = AndroidInitializationSettings('app_icon');
  final IOSInitializationSettings initializationSettingsIOS = IOSInitializationSettings();
  final InitializationSettings initializationSettings = InitializationSettings(
    android: initializationSettingsAndroid,
    iOS: initializationSettingsIOS,
  );
  await flutterLocalNotificationsPlugin.initialize(initializationSettings);
}

Future<void> _showNotification(String title, String body) async {
  const AndroidNotificationDetails androidPlatformChannelSpecifics = AndroidNotificationDetails(
    'r3al3r_channel', 'R3AL3R Notifications', 'Notifications for R3AL3R AI',
    importance: Importance.max,
    priority: Priority.high,
  );
  const NotificationDetails platformChannelSpecifics = NotificationDetails(android: androidPlatformChannelSpecifics);
  await flutterLocalNotificationsPlugin.show(0, title, body, platformChannelSpecifics);
}

String sanitizeInput(String input) {
  return input.replaceAll(RegExp(r'[<>]'), '');
}

class R3AL3RApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'R3AL3R AI',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      initialRoute: '/login',
      routes: {
        '/login': (context) => LoginScreen(),
        '/dashboard': (context) => DashboardScreen(),
        '/query': (context) => QueryScreen(),
      },
    );
  }
}

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _userIdController = TextEditingController();
  final TextEditingController _soulKeyController = TextEditingController();
  String _error = '';
  final storage = FlutterSecureStorage();
  final SpeechToText _speech = SpeechToText();
  bool _speechEnabled = false;

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  void _initSpeech() async {
    _speechEnabled = await _speech.initialize();
    setState(() {});
  }

  Future<void> _login() async {
    final userId = sanitizeInput(_userIdController.text);
    final soulKey = sanitizeInput(_soulKeyController.text);
    final soulKeyHash = sha256.convert(utf8.encode(soulKey)).toString();
    final key = encrypt.Key.fromUtf8(soulKey.padRight(32, '0'));
    final encrypter = encrypt.Encrypter(encrypt.AES(key));
    final encrypted = encrypter.encrypt(userId, iv: encrypt.IV.fromLength(16));

    try {
      final response = await http.post(
        Uri.parse('https://api.r3al3r.ai/api/transfer'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'soul_key': encrypted.base64,
          'ethical': true,
        }),
      );
      if (response.statusCode == 200) {
        final token = jsonDecode(response.body)['token'];
        final refreshToken = jsonDecode(response.body)['refresh_token'];
        if (!JwtDecoder.isExpired(token)) {
          await storage.write(key: 'token', value: token);
          await storage.write(key: 'refresh_token', value: refreshToken);
          await storage.write(key: 'user_id', value: userId);
          Navigator.pushReplacementNamed(context, '/dashboard', arguments: {'token': token, 'user_id': userId});
        } else {
          setState(() => _error = 'Invalid or expired token');
        }
      } else {
        setState(() => _error = 'Login failed');
      }
    } catch (e) {
      setState(() => _error = 'Error: $e');
    }
  }

  Future<void> _startListening() async {
    if (_speechEnabled) {
      await _speech.listen(onResult: (result) {
        setState(() {
          _soulKeyController.text = result.recognizedWords;
        });
      });
    } else {
      setState(() => _error = 'Speech recognition not available');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('R3AL3R AI Login')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TailwindTextField(
              controller: _userIdController,
              label: 'User ID',
              placeholder: 'Enter your user ID',
            ),
            TailwindTextField(
              controller: _soulKeyController,
              label: 'Soul Key',
              placeholder: 'Enter your soul key',
              obscureText: true,
            ),
            TailwindButton(
              onPressed: _login,
              child: Text('Login'),
            ),
            TailwindButton(
              onPressed: _startListening,
              child: Text('Speak Soul Key'),
            ),
            if (_error.isNotEmpty)
              Text(_error, style: TextStyle(color: Colors.red)),
          ],
        ),
      ),
    );
  }
}

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final args = ModalRoute.of(context)!.settings.arguments as Map;
    final token = args['token'];
    final userId = args['user_id'];
    return Scaffold(
      appBar: AppBar(title: Text('R3AL3R Dashboard')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Welcome, $userId!'),
            TailwindButton(
              onPressed: () => Navigator.pushNamed(context, '/query', arguments: {'token': token, 'user_id': userId}),
              child: Text('Make a Query'),
            ),
          ],
        ),
      ),
    );
  }
}

class QueryScreen extends StatefulWidget {
  @override
  _QueryScreenState createState() => _QueryScreenState();
}

class _QueryScreenState extends State<QueryScreen> {
  final TextEditingController _queryController = TextEditingController();
  String _response = '';
  final storage = FlutterSecureStorage();

  Future<void> _submitQuery() async {
    final args = ModalRoute.of(context)!.settings.arguments as Map;
    final token = args['token'];
    final userId = args['user_id'];
    final query = sanitizeInput(_queryController.text);

    try {
      if (JwtDecoder.isExpired(token)) await _refreshToken();
      final response = await http.post(
        Uri.parse('https://api.r3al3r.ai/api/query_anything'),
        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer $token'},
        body: jsonEncode({'user_id': userId, 'query': query}),
      );
      if (response.statusCode == 200) {
        setState(() => _response = jsonDecode(response.body)['response_id']);
        await _showNotification('Query Submitted', 'Your query is queued for ethical review');
      } else {
        setState(() => _response = 'Error: ${jsonDecode(response.body)['error']}');
      }
    } catch (e) {
      setState(() => _response = 'Error: $e');
    }
  }

  Future<void> _refreshToken() async {
    final refreshToken = await storage.read(key: 'refresh_token');
    if (refreshToken != null) {
      final response = await http.post(
        Uri.parse('https://api.r3al3r.ai/api/refresh_token'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'refresh_token': refreshToken}),
      );
      if (response.statusCode == 200) {
        final newToken = jsonDecode(response.body)['token'];
        await storage.write(key: 'token', value: newToken);
        return;
      }
    }
    Navigator.pushReplacementNamed(context, '/login');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Query R3AL3R')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TailwindTextField(
              controller: _queryController,
              label: 'Query',
              placeholder: 'Ask anything...',
            ),
            TailwindButton(
              onPressed: _submitQuery,
              child: Text('Submit Query'),
            ),
            if (_response.isNotEmpty)
              Text(_response, style: TextStyle(color: _response.contains('Error') ? Colors.red : Colors.green)),
          ],
        ),
      ),
    );
  }
}
