package com.example.raulbarbosa.stresssave;

import android.accounts.Account;
import android.accounts.AccountManager;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.IBinder;
import android.util.Patterns;
import android.widget.Toast;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.regex.Pattern;

public class MyService extends Service {
    private DatabaseReference mDatabase;
    public static double REFERENCE = 0.00002;
    Double db;

    private void writeNewUser(Double db) {
        mDatabase = FirebaseDatabase.getInstance().getReference();
        String id = mDatabase.push().getKey();
        Decibel user = new Decibel(db);
        mDatabase.child(id).setValue(user);
        Toast.makeText(this, "Sucess", Toast.LENGTH_LONG).show();
    }
    @Override
    public IBinder onBind(Intent intent) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void onCreate() {
        //Toast.makeText(this, " MyService Created ", Toast.LENGTH_LONG).show();
    }

    @Override
    public void onStart(Intent intent, int startId) {
        Toast.makeText(this, "Deu", Toast.LENGTH_LONG).show();
        mDatabase = FirebaseDatabase.getInstance().getReference();
        db=getNoiseLevel();
        // s=Double.toString(db);
        writeNewUser(db);

    }

    @Override
    public void onDestroy() {
        // TODO Auto-generated method stub
        Toast.makeText(this, "Services Stopped", Toast.LENGTH_SHORT).show();
        super.onDestroy();
    }





    public double getNoiseLevel()
    {
        int bufferSize = AudioRecord.getMinBufferSize(44100,AudioFormat.CHANNEL_IN_DEFAULT,AudioFormat.ENCODING_PCM_16BIT);
        //making the buffer bigger....
        bufferSize=bufferSize*4;
        AudioRecord recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                44100, AudioFormat.CHANNEL_IN_DEFAULT, AudioFormat.ENCODING_PCM_16BIT, bufferSize);

        short data [] = new short[bufferSize];
        double average = 0.0;
        recorder.startRecording();
        //recording data;
        recorder.read(data, 0, bufferSize);

        recorder.stop();
        for (short s : data)
        {
            if(s>0)
            {
                average += Math.abs(s);
            }
            else
            {
                bufferSize--;
            }
        }
        //x=max;
        double x = average/bufferSize;
        recorder.release();
        double db=0;

        // calculating the pascal pressure based on the idea that the max amplitude (between 0 and 32767) is
        // relative to the pressure
        double pressure = x/51805.5336; //the value 51805.5336 can be derived from asuming that x=32767=0.6325 Pa and x=1 = 0.00002 Pa (the reference value)
        db = (20 * Math.log10(pressure/REFERENCE));
        if(db>0)
        {
            return db;
        }
        return db;
    }


}