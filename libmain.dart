import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart';

void main() {
  runApp(R3AL3RApp());
}

class R3AL3RApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'R3ÆLƎR AI',
      theme: ThemeData(
        primaryColor: Colors.green,
        scaffoldBackgroundColor: Colors.black,
        textTheme: TextTheme(
          bodyText2: TextStyle(color: Colors.green),
          button: TextStyle(color: Colors.black),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            primary: Colors.green,
            onPrimary: Colors.black,
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(),
          filled: true,
          fillColor: Colors.black54,
          labelStyle: TextStyle(color: Colors.green),
        ),
      ),
      home: LoginScreen(),
    );
  }
}

class MatrixBackground extends StatefulWidget {
  final Widget child;
  MatrixBackground({required this.child});

  @override
  _MatrixBackgroundState createState() => _MatrixBackgroundState();
}

class _MatrixBackgroundState extends State<MatrixBackground> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  List<int> drops = [];
  double fontSize = 14;
  List<String> facePattern = [
    '      1111      ',
    '    11    11    ',
    '   1  00  00 1  ',
    '   1 0  0 0  1  ',
    '    11 0 0 11   ',
    '      1111      '
  ];
  int faceProgress = 0;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 33),
    )..addListener(() {
        setState(() {});
      });
    _controller.repeat();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      setState(() {
        drops = List.filled((MediaQuery.of(context).size.width / fontSize).floor(), 1);
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        CustomPaint(
          painter: MatrixPainter(drops, fontSize, facePattern, faceProgress, () {
            setState(() {
              faceProgress++;
            });
          }),
          size: Size.infinite,
        ),
        widget.child,
      ],
    );
  }
}

class MatrixPainter extends CustomPainter {
  final List<int> drops;
  final double fontSize;
  final List<String> facePattern;
  int faceProgress;
  final VoidCallback onFaceProgress;

  MatrixPainter(this.drops, this.fontSize, this.facePattern, this.faceProgress, this.onFaceProgress);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.black.withOpacity(0.05);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), paint);
    final textPaint = TextPainter(textDirection: TextDirection.ltr);
    final chars = '01';
    final faceX = size.width / 2 - (facePattern[0].length * fontSize) / 2;
    final faceY = size.height / 2 - (facePattern.length * fontSize) / 2;

    for (int i = 0; i < drops.length; i++) {
      final x = i * fontSize;
      final y = drops[i] * fontSize;
      bool isFacePixel = false;

      if (faceProgress < facePattern.length * facePattern[0].length) {
        final faceRow = faceProgress ~/ facePattern[0].length;
        final faceCol = faceProgress % facePattern[0].length;
        final faceChar = facePattern[faceRow][faceCol];
        if (faceChar == '1' || faceChar == '0') {
          final facePixelX = faceX + faceCol * fontSize;
          final facePixelY = faceY + faceRow * fontSize;
          if ((x - facePixelX).abs() < fontSize && (y - facePixelY).abs() < fontSize) {
            textPaint.text = TextSpan(
              text: faceChar,
              style: TextStyle(color: Colors.green, fontSize: fontSize, fontFamily: 'monospace'),
            );
            textPaint.layout();
            textPaint.paint(canvas, Offset(x, y));
            isFacePixel = true;
            faceProgress++;
            onFaceProgress();
          }
        }
      }

      if (!isFacePixel) {
        final text = chars[Random().nextInt(chars.length)];
        textPaint.text = TextSpan(
          text: text,
          style: TextStyle(color: Colors.green, fontSize: fontSize, fontFamily: 'monospace'),
        );
        textPaint.layout();
        textPaint.paint(canvas, Offset(x, y));
      }

      if (y > size.height && Random().nextDouble() > 0.975) {
        drops[i] = 0;
      } else {
        drops[i]++;
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _userIdController = TextEditingController();
  String token = '';

  Future<void> login() async {
    final response = await http.post(
      Uri.parse('ip-172-31-38-60.us-east-2.compute.internal'), // Replace with server IP
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'user_id': _userIdController.text}),
    );
    final data = jsonDecode(response.body);
    if (data['token'] != null) {
      setState(() {
        token = data['token'];
      });
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => MainScreen(token: token)),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Login failed: ${data['error']}')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: MatrixBackground(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('R3ÆLƎR AI', style: TextStyle(fontSize: 32, color: Colors.green)),
              SizedBox(height: 20),
              TextField(
                controller: _userIdController,
                decoration: InputDecoration(labelText: 'User ID'),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: login,
                child: Text('Login'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  final String token;
  MainScreen({required this.token});

  @override
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final _inputDataController = TextEditingController();
  final _languageController = TextEditingController();
  final _taskController = TextEditingController();
  String _taskType = 'analysis';
  String _output = '';
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

  Future<void> _startListening() async {
    if (_speechEnabled) {
      await _speech.listen(
        onResult: (result) {
          setState(() {
            _taskType = result.recognizedWords.toLowerCase();
          });
        },
      );
    }
  }

  Future<void> generateInsight() async {
    final response = await http.post(
      Uri.parse('ip-172-31-38-60.us-east-2.compute.internal/'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.token}',
      },
      body: jsonEncode({
        'input_data': _inputDataController.text,
        'task_type': _taskType,
      }),
    );
    final data = jsonDecode(response.body);
    setState(() {
      _output = data['insight'] ?? data['error'];
    });
  }

  Future<void> generateCode() async {
    final response = await http.post(
      Uri.parse('ip-172-31-38-60.us-east-2.compute.internal/api/code_generate'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.token}',
      },
      body: jsonEncode({
        'language': _languageController.text,
        'task': _taskController.text,
      }),
    );
    final data = jsonDecode(response.body);
    setState(() {
      _output = data['code'] ?? data['error'];
    });
  }

  Future<void> subscribe() async {
    final response = await http.post(
      Uri.parse('http://your-server:5000/api/subscribe'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.token}',
      },
      body: jsonEncode({'plan': 'monthly'}),
    );
    final data = jsonDecode(response.body);
    setState(() {
      _output = data['message'] ?? data['error'];
    });
  }

  Future<void> getInsights() async {
    final response = await http.get(
      Uri.parse('http://your-server:5000/api/insights'),
      headers: {'Authorization': 'Bearer ${widget.token}'},
    );
    final data = jsonDecode(response.body);
    setState(() {
      _output = jsonEncode(data['insights'] ?? data['error'], indent: 2);
    });
  }

  Future<void> downloadApk() async {
    // APK download not implemented in app; use website
    setState(() {
      _output = 'Please download APK from the website.';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: MatrixBackground(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('R3ÆLƎR AI', style: TextStyle(fontSize: 32)),
              SizedBox(height: 20),
              TextField(
                controller: _inputDataController,
                decoration: InputDecoration(labelText: 'Input Data (CSV path)'),
              ),
              DropdownButton<String>(
                value: _taskType,
                items: ['analysis', 'prediction', 'ideation', 'pattern', 'treadmill']
                    .map((value) => DropdownMenuItem(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) => setState(() {
                  _taskType = value!;
                }),
              ),
              ElevatedButton(
                onPressed: generateInsight,
                child: Text('Generate Insight'),
              ),
              TextField(
                controller: _languageController,
                decoration: InputDecoration(labelText: 'Language (e.g., Python)'),
              ),
              TextField(
                controller: _taskController,
                decoration: InputDecoration(labelText: 'Task (e.g., print hello)'),
              ),
              ElevatedButton(
                onPressed: generateCode,
                child: Text('Generate Code'),
              ),
              ElevatedButton(
                onPressed: subscribe,
                child: Text('Subscribe'),
              ),
              ElevatedButton(
                onPressed: getInsights,
                child: Text('View Insights'),
              ),
              ElevatedButton(
                onPressed: downloadApk,
                child: Text('Download APK'),
              ),
              ElevatedButton(
                onPressed: _speechEnabled ? _startListening : null,
                child: Text('Voice Input'),
              ),
              SizedBox(height: 20),
              Text(_output, style: TextStyle(fontFamily: 'monospace')),
            ],
          ),
        ),
      ),
    );
  }
}
