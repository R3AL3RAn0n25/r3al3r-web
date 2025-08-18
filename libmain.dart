import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:speech_to_text/speech_to_text.dart' as stt;

void main() {
  runApp(R3AL3RApp());
}

class R3AL3RApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'R3ÆLƎR AI',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController _userIdController = TextEditingController();
  final TextEditingController _inputDataController = TextEditingController();
  String _insight = '';
  String _token = '';
  String _selectedTask = 'analysis';
  stt.SpeechToText _speech = stt.SpeechToText();
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
    final userId = _userIdController.text;
    try {
      final response = await http.post(
        Uri.parse('http://your-server:5000/api/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': userId}),
      );
      if (response.statusCode == 200) {
        setState(() {
          _token = jsonDecode(response.body)['token'];
          _insight = 'Logged in!';
        });
      } else {
        setState(() {
          _insight = 'Error: ${jsonDecode(response.body)['error']}';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error logging in';
      });
    }
  }

  Future<void> _getInsight() async {
    final inputData = _inputDataController.text.isEmpty ? null : _inputDataController.text;
    try {
      final response = await http.post(
        Uri.parse('http://your-server:5000/api/insight'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $_token',
        },
        body: jsonEncode({'task_type': _selectedTask, 'input_data': inputData}),
      );
      if (response.statusCode == 200) {
        setState(() {
          _insight = jsonDecode(response.body)['insight'];
        });
      } else {
        setState(() {
          _insight = 'Error: ${jsonDecode(response.body)['error']}';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error generating insight';
      });
    }
  }

  Future<void> _subscribe() async {
    try {
      final response = await http.post(
        Uri.parse('http://your-server:5000/api/subscribe'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $_token',
        },
        body: jsonEncode({'plan': 'monthly'}),
      );
      if (response.statusCode == 200) {
        setState(() {
          _insight = jsonDecode(response.body)['message'];
        });
      } else {
        setState(() {
          _insight = 'Error: ${jsonDecode(response.body)['error']}';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error subscribing';
      });
    }
  }

  Future<void> _getInsights() async {
    try {
      final response = await http.get(
        Uri.parse('http://your-server:5000/api/insights'),
        headers: {'Authorization': 'Bearer $_token'},
      );
      if (response.statusCode == 200) {
        setState(() {
          _insight = jsonDecode(response.body)['insights'].join('\n');
        });
      } else {
        setState(() {
          _insight = 'Error: ${jsonDecode(response.body)['error']}';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error retrieving insights';
      });
    }
  }

  Future<void> _adaptData() async {
    final inputData = _inputDataController.text.isEmpty ? null : _inputDataController.text;
    try {
      final response = await http.post(
        Uri.parse('http://your-server:5000/api/adapt'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $_token',
        },
        body: jsonEncode({'data': inputData}),
      );
      if (response.statusCode == 200) {
        setState(() {
          _insight = jsonDecode(response.body)['message'];
        });
      } else {
        setState(() {
          _insight = 'Error: ${jsonDecode(response.body)['error']}';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error adapting data';
      });
    }
  }

  Future<void> _getAffidavit() async {
    try {
      final response = await http.get(
        Uri.parse('http://your-server:5000/api/affidavit'),
      );
      if (response.statusCode == 200) {
        setState(() {
          _insight = jsonDecode(response.body)['affidavit'];
        });
      } else {
        setState(() {
          _insight = 'Error retrieving affidavit';
        });
      }
    } catch (e) {
      setState(() {
        _insight = 'Error retrieving affidavit';
      });
    }
  }

  Future<void> _startListening() async {
    if (_speechEnabled) {
      await _speech.listen(onResult: (result) {
        setState(() {
          _inputDataController.text = result.recognizedWords;
        });
      });
    } else {
      setState(() {
        _insight = 'Speech recognition not available';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('R3ÆLƎR AI')),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _userIdController,
              decoration: InputDecoration(labelText: 'User ID'),
            ),
            DropdownButton<String>(
              value: _selectedTask,
              items: ['analysis', 'prediction', 'ideation', 'pattern']
                  .map((task) => DropdownMenuItem(
                        value: task,
                        child: Text(task),
                      ))
                  .toList(),
              onChanged: (value) {
                setState(() {
                  _selectedTask = value!;
                });
              },
            ),
            TextField(
              controller: _inputDataController,
              decoration: InputDecoration(labelText: 'CSV Path (optional)'),
            ),
            ElevatedButton(
              onPressed: _login,
              child: Text('Login'),
            ),
            ElevatedButton(
              onPressed: _getInsight,
              child: Text('Get Insight'),
            ),
            ElevatedButton(
              onPressed: _subscribe,
              child: Text('Subscribe'),
            ),
            ElevatedButton(
              onPressed: _getInsights,
              child: Text('View Insights'),
            ),
            ElevatedButton(
              onPressed: _adaptData,
              child: Text('Adapt Data'),
            ),
            ElevatedButton(
              onPressed: _getAffidavit,
              child: Text('Get Affidavit'),
            ),
            ElevatedButton(
              onPressed: _startListening,
              child: Text('Speak Input'),
            ),
            SizedBox(height: 20),
            Text('Output: $_insight'),
          ],
        ),
      ),
    );
  }
}