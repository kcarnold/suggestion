import React from 'react';

class Consent extends React.Component {
  render() {
    return (

      <section className="container" id="Other" style={{marginBottom: 15, padding: '10px 10px', fontFamily: 'Verdana, Geneva, sans-serif', color: '#333333', fontSize: '0.9em'}}>
        <p>We are testing apps to help people write, and we need your help to try out writing with some of these apps. (It's actually just a webpage to visit on your phone, nothing to install.)</p>
        <p><b>Eligibility</b><br />
          Participants should have normal or corrected-to-normal vision and experience entering text on a touchscreen device. <b>You will need a smartphone with WiFi to complete this task.</b> Participants must be adults (at least 18 years old). <strong>If you have previously completed this study, please don't do it again. If you realize partway through, that's ok, finish it and leave a note.</strong></p>
        <p>Your phone will need to have a good connection with our server for this task.</p>

        <table cellPadding={0} cellSpacing={0} style={{marginLeft: '5.4pt', borderCollapse: 'collapse', border: 'none'}}>
          <tbody>
            <tr>
              <td style={{border: 'solid black 1.5pt', borderBottom: 'solid silver 1.5pt', padding: '0in 2.15pt 0in 2.9pt'}}>
                <p><span style={{fontSize: '14.0pt'}}>Study Title: Predictive typing with long suggestions</span></p>
              </td>
            </tr>
            <tr>
              <td style={{border: 'solid black 1.5pt', borderTop: 'none', padding: '0in 2.15pt 0in 2.9pt'}}>
                <p><span style={{fontSize: '14.0pt'}}>Researcher: Kenneth C. Arnold</span></p>
              </td>
            </tr>
          </tbody>
        </table>
        <p><b><span style={{fontSize: '14.0pt'}}>Participation is voluntary</span></b></p>
        <p>It is your choice whether or not to participate in this research.&nbsp; If you choose to participate, you may change your mind and leave the study at any time by returning the HIT.&nbsp; Refusal to participate or stopping your participation will involve no penalty or loss of benefits to which you are otherwise entitled. Participants must be adults 18+.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>What is the purpose of this research?</span></b></p>
        <p>We are studying systems that make suggestions while someone is typing to find out how the system's design affects the writing process, writing outcome, and the author's subjective experience, and how psychological factors may influence those effects.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>What can I expect if I take part in this research? What is the time commitment?</span></b></p>
        <p>You will type a few paragraphs of text, using keyboards that give various kinds of help in the process. You should not include any personally identifiable information in what you write.</p>
        <p>There will be questionnaires throughout the study, which may include questions about your age, gender, personality, computer use, and other demographic information.</p>
        <p>Some aspects of this study will occur at your own pace, so the time commitment may vary, but we expect it to be <strong>between 15 and 30 minutes</strong>. There will be an opportunity to take a break about every 5 minutes.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>What are the risks and possible discomforts?</span></b></p>
        <p>There are no anticipated risks beyond normal use of a computer or smartphone. Please take a break if you start to feel any discomfort from prolonged computer use.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>Are there any benefits from being in this research study? </span></b></p>
        <p>We cannot promise any benefits to you or others from your taking part in this research. The results of this research may inform future advances in computer systems that assist writing.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>Will I be compensated for participating in this research?</span></b></p>
        <p>The target payment for this task is $9/hr. Your payment will in no way depend on how you use the unusual features of the app being tested, if any. The reward for a specific HIT will be determined based on the estimated time to complete the task at $9/hr.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>If I take part in this research, how will my privacy be protected? What happens to the information you collect? </span></b></p>
        <p>We will not record any personally identifying information. De-identified data may be shared with other researchers and other participants in this study.</p>
        <p>The MTurk platform provides access to your worker ID, which in some cases can be mapped to your name and work history. We are relying on the security of that platform to maintain your confidentiality. To partially mitigate the risk of re-identification, we will assign you a random identifier specific to this study and delete the mapping between this identifier and your worker ID 6 months after the experiment concludes. But if the security of the MTurk platform or our account is breached, it may be possible to re-identify your work, as with any MTurk task. <b>Please make sure to mark your Amazon Profile as private</b> if you do not want it to be found from your Mechanical Turk Worker ID.</p>
        <p><b><span style={{fontSize: '14.0pt'}}>If I have any questions, concerns or complaints about this research study, who can I talk to?</span></b></p>
        <p>The researcher for this study is Kenneth C. Arnold<i> </i>who can be reached at<i> </i>kcarnold@seas.harvard.edu, 617-299-6536, or 33 Oxford St MD 240, Cambridge MA 02138.<i> </i>The faculty sponsor is Krzysztof Z. Gajos who can be reached at kgajos@seas.harvard.edu</p>
        <ul style={{marginTop: '0in'}} type="disc">
          <li>If you have questions, concerns, or complaints,</li>
          <li>If you would like to talk to the research team,</li>
          <li>If you think the research has harmed you, or</li>
          <li>If you wish to withdraw from the study.</li>
        </ul>
        <p>This research has been reviewed by the Committee on the Use of Human Subjects in Research at Harvard University.&nbsp; They can be reached at 617-496-2847, 1414 Massachusetts Avenue, Second Floor, Cambridge, MA 02138, or cuhs@fas.harvard.edu for any of the following:</p>
        <ul style={{marginTop: '0in'}} type="disc">
          <li>If your questions, concerns, or complaints are not being answered by the research team,</li>
          <li>If you cannot reach the research team,</li>
          <li>If you want to talk to someone besides the research team, or</li>
          <li>If you have questions about your rights as a research participant.</li>
        </ul>
      </section>
    );
  }
}

export default Consent;
